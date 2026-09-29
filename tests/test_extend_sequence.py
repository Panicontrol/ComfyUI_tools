import pytest
import torch

from comfyui_tools.image_nodes import ExtendSequence, extend_indices


def numbered(frames, height=2, width=3):
    """A batch whose every pixel holds its own frame index."""
    return torch.arange(frames, dtype=torch.float32).view(frames, 1, 1, 1).expand(
        frames, height, width, 3
    ).clone()


def frame_ids(images):
    return [int(v) for v in images[:, 0, 0, 0].tolist()]


def test_the_minimax_case_98_to_124():
    indices = extend_indices(98, 124, "mirror")
    assert len(indices) == 124
    assert indices[:98] == list(range(98))
    # the extra 26 frames mirror back from the second-to-last frame
    assert indices[98:] == list(range(96, 70, -1))


def test_mirror_does_not_repeat_the_turning_frames():
    assert extend_indices(4, 12, "mirror") == [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1]


def test_mirror_keeps_going_past_several_turns():
    indices = extend_indices(3, 20, "mirror")
    assert len(indices) == 20
    assert all(abs(a - b) == 1 for a, b in zip(indices, indices[1:]))


def test_mirror_of_a_single_frame_holds_it():
    assert extend_indices(1, 5, "mirror") == [0] * 5


def test_loop():
    assert extend_indices(3, 8, "loop") == [0, 1, 2, 0, 1, 2, 0, 1]


def test_hold_last():
    assert extend_indices(3, 6, "hold_last") == [0, 1, 2, 2, 2, 2]


def test_a_long_enough_sequence_is_kept_whole():
    assert extend_indices(150, 124, "mirror") == list(range(150))
    assert extend_indices(124, 124, "mirror") == list(range(124))


def test_trim_longer_cuts_to_the_exact_length():
    assert extend_indices(150, 124, "mirror", trim_longer=True) == list(range(124))


def test_invalid_lengths():
    with pytest.raises(ValueError):
        extend_indices(0, 10, "mirror")
    with pytest.raises(ValueError):
        extend_indices(10, 0, "mirror")


def test_node_extends_images_in_mirror_order():
    images, masks, count, order = ExtendSequence().extend(numbered(4), 7, "mirror", False)
    assert images.shape == (7, 2, 3, 3)
    assert frame_ids(images) == [0, 1, 2, 3, 2, 1, 0]
    assert count == 7
    assert order == "0,1,2,3,2,1,0"
    assert masks.shape == (7, 2, 3)
    assert masks.max() == 0.0  # no masks given


def test_node_carries_masks_in_the_same_order():
    masks = torch.arange(4, dtype=torch.float32).view(4, 1, 1).expand(4, 2, 3).clone() / 10
    _, out_masks, _, _ = ExtendSequence().extend(numbered(4), 6, "mirror", False, masks=masks)
    assert [round(v * 10) for v in out_masks[:, 0, 0].tolist()] == [0, 1, 2, 3, 2, 1]


def test_a_single_mask_is_used_for_every_frame():
    mask = torch.ones((1, 2, 3))
    _, out_masks, _, _ = ExtendSequence().extend(numbered(3), 5, "loop", False, masks=mask)
    assert out_masks.shape == (5, 2, 3)
    assert out_masks.min() == 1.0


def test_a_2d_mask_is_accepted():
    _, out_masks, _, _ = ExtendSequence().extend(numbered(2), 3, "loop", False, masks=torch.ones((2, 3)))
    assert out_masks.shape == (3, 2, 3)


def test_mismatched_masks_are_an_error():
    with pytest.raises(ValueError, match="masks have 2 frames"):
        ExtendSequence().extend(numbered(4), 8, "mirror", False, masks=torch.zeros((2, 2, 3)))


def test_node_leaves_a_long_batch_alone_unless_asked():
    images, _, count, _ = ExtendSequence().extend(numbered(10), 6, "mirror", False)
    assert count == 10 and frame_ids(images) == list(range(10))
    images, _, count, _ = ExtendSequence().extend(numbered(10), 6, "mirror", True)
    assert count == 6 and frame_ids(images) == list(range(6))
