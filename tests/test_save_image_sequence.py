import pytest
import torch

from comfyui_tools.image_nodes import LoadImageSequence, SaveImageSequence


def save(images, folder, **kwargs):
    params = dict(new_folder="depth", number_version=4, start_frame_index=1001,
                  padding_frame_name=5, overwrite=False)
    params.update(kwargs)
    return SaveImageSequence().save(images, str(folder), **params)["result"]


def test_layout_and_round_trip_with_masks(tmp_path):
    images, masks = torch.rand(2, 6, 8, 3), torch.rand(2, 6, 8)
    folder, version, count = save(images, tmp_path, masks=masks)
    assert folder == str(tmp_path / "depth_v004") and (version, count) == (4, 2)

    loaded, loaded_masks, _, names = LoadImageSequence().load(folder, "*.png", "name", False, 0, 1, 0, "error")
    assert names.splitlines() == ["depth_v004.01001.png", "depth_v004.01002.png"]
    assert torch.allclose(loaded, images, atol=1 / 255)
    assert torch.allclose(loaded_masks, masks, atol=1 / 255)


def test_existing_frames_are_kept_unless_overwrite(tmp_path):
    images = torch.rand(1, 4, 4, 3)
    save(images, tmp_path)
    with pytest.raises(ValueError, match="raise number_version"):
        save(images, tmp_path)
    save(images, tmp_path, overwrite=True)
