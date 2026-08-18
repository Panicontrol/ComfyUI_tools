import pytest
import torch

from comfyui_tools.image_nodes import ImageInfo, ImagePadToRatio, ImageResize


def test_resize_longest_side(image):
    result, width, height = ImageResize().resize(
        image, "longest_side", 128, 1.0, 0, 0, True, 8, "bilinear"
    )
    assert (width, height) == (64, 128)  # source is 32x64, portrait
    assert result.shape == (2, 128, 64, 3)


def test_resize_shortest_side(image):
    _, width, height = ImageResize().resize(
        image, "shortest_side", 64, 1.0, 0, 0, True, 8, "bilinear"
    )
    assert (width, height) == (64, 128)


def test_resize_megapixels_keeps_aspect(image):
    _, width, height = ImageResize().resize(
        image, "megapixels", 0, 0.25, 0, 0, True, 1, "bilinear"
    )
    assert width * height == pytest.approx(250_000, rel=0.02)
    assert width / height == pytest.approx(0.5, rel=0.02)


def test_resize_width_height_fits_inside_the_box(image):
    _, width, height = ImageResize().resize(
        image, "width_height", 0, 1.0, 100, 100, True, 1, "bilinear"
    )
    assert (width, height) == (50, 100)


def test_resize_width_height_exact_when_aspect_is_ignored(image):
    _, width, height = ImageResize().resize(
        image, "width_height", 0, 1.0, 96, 40, False, 8, "bilinear"
    )
    assert (width, height) == (96, 40)


def test_resize_rounds_to_multiple(image):
    _, width, height = ImageResize().resize(
        image, "longest_side", 100, 1.0, 0, 0, True, 16, "bilinear"
    )
    assert width % 16 == 0 and height % 16 == 0


def test_pad_to_ratio_centers_and_marks_padding(image):
    padded, mask, width, height = ImagePadToRatio().pad(image, 1.0, 1.0, "center", "#000000", 8)
    assert (width, height) == (64, 64)
    assert padded.shape == (2, 64, 64, 3)
    # the original pixels survive untouched in the middle of the canvas
    assert torch.equal(padded[:, :, 16:48, :], image)
    # mask marks only the padded columns
    assert mask[:, :, 16:48].max() == 0.0
    assert mask[:, :, :16].min() == 1.0


def test_pad_to_ratio_respects_position(image):
    padded, mask, _, _ = ImagePadToRatio().pad(image, 1.0, 1.0, "left", "#ffffff", 8)
    assert torch.equal(padded[:, :, :32, :], image)
    assert mask[:, :, :32].max() == 0.0
    assert padded[0, 0, 40, 0] == pytest.approx(1.0)


def test_pad_never_shrinks_an_image(image):
    _, _, width, height = ImagePadToRatio().pad(image, 1.0, 4.0, "center", "#000000", 8)
    assert width >= 32 and height >= 64


def test_image_info(image):
    width, height, batch, aspect, summary = ImageInfo().info(image)
    assert (width, height, batch) == (32, 64, 2)
    assert aspect == pytest.approx(0.5)
    assert summary.startswith("32x64 | batch 2")
