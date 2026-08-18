import pytest
import torch

from comfyui_tools.mask_nodes import MaskBoundingBox, MaskCombine, MaskGrowFeather


def test_grow_expands_the_filled_area(mask):
    grown, = MaskGrowFeather().process(mask, 2, 0, False)
    assert grown[0, 2, 2] == pytest.approx(1.0)
    assert grown.sum() > mask.sum()


def test_shrink_reduces_the_filled_area(mask):
    shrunk, = MaskGrowFeather().process(mask, -1, 0, False)
    assert shrunk.sum() < mask.sum()
    assert shrunk[0, 4, 4] == pytest.approx(0.0)


def test_feather_softens_the_edge(mask):
    feathered, = MaskGrowFeather().process(mask, 0, 3, False)
    edge = feathered[0, 3, 5].item()
    assert 0.0 < edge < 1.0
    assert feathered.max() <= 1.0


def test_invert(mask):
    inverted, = MaskGrowFeather().process(mask, 0, 0, True)
    assert torch.allclose(inverted, 1.0 - mask)


def test_combine_operations(mask):
    other = torch.zeros_like(mask)
    other[:, 6:10, 6:10] = 1.0

    union, = MaskCombine().combine(mask, other, "union", 1.0)
    intersection, = MaskCombine().combine(mask, other, "intersection", 1.0)
    difference, = MaskCombine().combine(mask, other, "difference", 1.0)
    xor, = MaskCombine().combine(mask, other, "xor", 1.0)

    assert union.sum() == pytest.approx(4 * 4 * 2 - 2 * 2)
    assert intersection.sum() == pytest.approx(2 * 2)
    assert difference.sum() == pytest.approx(4 * 4 - 2 * 2)
    assert xor.sum() == pytest.approx(2 * (4 * 4 - 2 * 2))


def test_combine_strength_scales_the_second_mask(mask):
    result, = MaskCombine().combine(torch.zeros_like(mask), mask, "add", 0.5)
    assert result.max() == pytest.approx(0.5)


def test_combine_resizes_mismatched_masks(mask):
    small = torch.ones((1, 8, 8))
    result, = MaskCombine().combine(mask, small, "union", 1.0)
    assert result.shape == mask.shape


def test_bounding_box(mask):
    x, y, width, height, coverage = MaskBoundingBox().bounds(mask, 0.05, 0)
    assert (x, y, width, height) == (4, 4, 4, 4)
    assert coverage == pytest.approx(16 / 256)


def test_bounding_box_padding_is_clamped_to_the_canvas(mask):
    x, y, width, height, _ = MaskBoundingBox().bounds(mask, 0.05, 10)
    assert (x, y) == (0, 0)
    assert (width, height) == (16, 16)


def test_bounding_box_of_an_empty_mask():
    empty = torch.zeros((1, 16, 16))
    assert MaskBoundingBox().bounds(empty, 0.05, 0) == (0, 0, 0, 0, 0.0)
