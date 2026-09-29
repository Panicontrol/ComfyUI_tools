import pytest
import torch

from comfyui_tools.image_nodes import ExtendSequence, RestoreSequenceLength


def numbered(frames):
    """A batch whose pixels hold their own frame index."""
    return torch.arange(frames, dtype=torch.float32).view(frames, 1, 1, 1).expand(frames, 2, 3, 3).clone()


@pytest.mark.parametrize("mode", ["mirror", "loop", "hold_last"])
def test_restore_undoes_extend(mode):
    original = numbered(98)
    extended = ExtendSequence().extend(original, 124, mode, False)[0]
    restored, _, count = RestoreSequenceLength().restore(extended, 0, original=original)
    assert count == 98 and torch.equal(restored, original)


def test_length_is_used_without_an_original():
    assert RestoreSequenceLength().restore(numbered(124), 98)[2] == 98


def test_a_short_batch_is_not_padded():
    assert RestoreSequenceLength().restore(numbered(90), 98)[2] == 90


def test_masks_are_cut_with_the_images():
    _, masks, _ = RestoreSequenceLength().restore(numbered(124), 98, masks=torch.zeros((124, 2, 3)))
    assert masks.shape == (98, 2, 3)
