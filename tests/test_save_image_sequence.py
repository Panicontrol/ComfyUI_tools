import zipfile

import av
import pytest
import torch

from comfyui_tools.image_nodes import LoadImageSequence, SaveImageSequence


def save(images, folder, **kwargs):
    params = dict(new_folder="depth", number_version=4, start_frame_index=1001,
                  padding_frame_name=5, overwrite=False)
    params.update(kwargs)
    return SaveImageSequence().save(images, str(folder), **params)["result"]


def test_video_sequence_and_zip_layout(tmp_path):
    images, masks = torch.rand(3, 6, 9, 3), torch.rand(3, 6, 9)  # odd width on purpose
    folder, version, count, video = save(images, tmp_path, masks=masks, need_zip=True)
    assert video == str(tmp_path / "depth_v004.mp4")
    assert folder == str(tmp_path / "depth_v004") and (version, count) == (4, 3)

    with av.open(video) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3
    with zipfile.ZipFile(tmp_path / "depth_v004.zip") as archive:
        assert archive.namelist()[0] == "depth_v004/depth_v004.01001.png"

    loaded, loaded_masks, _, names = LoadImageSequence().load(folder, "*.png", "name", False, 0, 1, 0, "error")
    assert names.splitlines()[0] == "depth_v004.01001.png"
    assert torch.allclose(loaded, images, atol=1 / 255)
    assert torch.allclose(loaded_masks, masks, atol=1 / 255)


def test_video_only(tmp_path):
    save(torch.rand(2, 4, 4, 3), tmp_path, need_save=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["depth_v004.mp4"]


def test_existing_files_are_kept_unless_overwrite(tmp_path):
    images = torch.rand(1, 4, 4, 3)
    save(images, tmp_path)
    with pytest.raises(ValueError, match="raise number_version"):
        save(images, tmp_path)
    save(images, tmp_path, overwrite=True)


def test_workflow_is_embedded_in_the_video(tmp_path):
    import json

    workflow = {"nodes": [{"id": 1, "type": "ToolsSaveImageSequence"}], "links": []}
    prompt = {"1": {"class_type": "ToolsSaveImageSequence", "inputs": {}}}
    *_, video = save(torch.rand(1, 4, 4, 3), tmp_path, need_save=False,
                     prompt=prompt, extra_pnginfo={"workflow": workflow})
    with av.open(video) as container:
        assert json.loads(container.metadata["workflow"]) == workflow
        assert json.loads(container.metadata["prompt"]) == prompt
