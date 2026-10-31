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
    folder, version, count, video, latent_path = save(images, tmp_path, masks=masks, need_zip=True)
    assert latent_path == ""  # nothing connected, nothing written
    assert video == str(tmp_path / "depth_v004.mp4")
    assert folder == str(tmp_path / "depth_v004") and (version, count) == (4, 3)

    with av.open(video) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3
    with zipfile.ZipFile(tmp_path / "depth_v004.zip") as archive:
        assert archive.namelist()[0] == "depth_v004/depth_v004.01001.png"

    loaded, loaded_masks, _, names = LoadImageSequence().load(folder, "*.png", "name", False, 0, 1, 0, "error")[:4]
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
    *_, video, _ = save(torch.rand(1, 4, 4, 3), tmp_path, need_save=False,
                     prompt=prompt, extra_pnginfo={"workflow": workflow})
    with av.open(video) as container:
        assert json.loads(container.metadata["workflow"]) == workflow
        assert json.loads(container.metadata["prompt"]) == prompt


def test_workflow_embedding_can_be_off_and_json_saved_instead(tmp_path):
    import json

    workflow = {"nodes": [{"id": 1}], "links": []}
    *_, video, _ = save(torch.rand(1, 4, 4, 3), tmp_path, need_save=False, embed_workflow=False,
                     save_workflow_json=True, prompt={"1": {}}, extra_pnginfo={"workflow": workflow})
    with av.open(video) as container:
        assert "workflow" not in container.metadata
    assert json.loads((tmp_path / "depth_v004.json").read_text()) == workflow


def test_latent_saved_next_to_the_video(tmp_path):
    from comfyui_tools.latent_nodes import LoadLatentFrom

    latent = {"samples": torch.randn(1, 24, 2, 4, 4)}
    *_, latent_path = save(torch.rand(1, 4, 4, 3), tmp_path, need_save=False, latent=latent)
    assert latent_path == str(tmp_path / "depth_v004.latent")
    assert torch.equal(LoadLatentFrom().load(latent_path)[0]["samples"], latent["samples"])

    (tmp_path / "depth_v004.mp4").unlink()  # only the latent is left: still refuses to overwrite it
    with pytest.raises(ValueError, match="depth_v004.latent"):
        save(torch.rand(1, 4, 4, 3), tmp_path, need_save=False, latent=latent)


def video_size(path):
    with av.open(path) as container:
        stream = container.streams.video[0]
        return stream.width, stream.height


def test_oversize_frames_get_a_preview_mp4_after_the_full_frames(tmp_path):
    images = torch.rand(2, 128, 400, 3)                # like 19584x6264 against 8192, scaled down
    folder, _, _, video, _ = save(images, tmp_path, video_max_side=164)
    assert video_size(video) == (164, 52)               # longer side 164, even sides
    from PIL import Image
    with Image.open(f"{folder}/depth_v004.01001.png") as frame:
        assert frame.size == (400, 128)                 # the frames stay full size
    assert not list(tmp_path.rglob("*.tmp"))

    folder, _, _, video, _ = save(images, tmp_path, number_version=5, video_max_side=164, video_oversize="skip")
    assert video == "" and not (tmp_path / "depth_v005.mp4").exists() and len(list((tmp_path / "depth_v005").iterdir())) == 2
    with pytest.raises(ValueError, match=r"400 px is over video_max_side 164 .*frames are saved in"):
        save(images, tmp_path, number_version=6, video_max_side=164, video_oversize="error")
    assert len(list((tmp_path / "depth_v006").iterdir())) == 2 and not (tmp_path / "depth_v006.mp4").exists()
    folder, _, _, video, _ = save(images, tmp_path, number_version=7, need_video=False)
    assert video == "" and not (tmp_path / "depth_v007.mp4").exists()


def test_a_failing_mp4_keeps_the_frames(tmp_path, monkeypatch):
    from comfyui_tools import image_nodes

    def broken(*args, **kwargs):
        raise RuntimeError('avcodec_open2("libx264") failed')

    monkeypatch.setattr(image_nodes, "write_mp4", broken)
    with pytest.raises(RuntimeError, match=r"frames are saved in .*depth_v004, but the mp4 failed"):
        save(torch.rand(2, 6, 8, 3), tmp_path)
    assert sorted(p.name for p in (tmp_path / "depth_v004").iterdir()) == ["depth_v004.01001.png", "depth_v004.01002.png"]


def test_jpg_tiff_and_validation(tmp_path):
    import numpy as np
    from PIL import Image, JpegImagePlugin

    images = torch.rand(1, 16, 24, 3)
    folder, *_ = save(images, tmp_path, format="jpg", jpg_quality=90, need_video=False)
    with Image.open(f"{folder}/depth_v004.01001.jpg") as jpg:
        assert JpegImagePlugin.get_sampling(jpg) == 0          # 4:4:4
        assert np.abs(np.asarray(jpg, np.float32) - images[0].numpy() * 255).mean() < 12

    assert SaveImageSequence.VALIDATE_INPUTS(format="jpg", bit_depth="16").startswith("jpg is 8-bit only")
    assert SaveImageSequence.VALIDATE_INPUTS(format="tiff", bit_depth="16") is True
    with pytest.raises(ValueError, match="jpg is 8-bit only"):
        save(images, tmp_path, number_version=5, format="jpg", bit_depth="16")

    folder, *_ = save(images, tmp_path, number_version=6, format="tiff", need_video=False)
    with Image.open(f"{folder}/depth_v006.01001.tif") as tif:
        assert np.array_equal(np.asarray(tif), (images[0] * 255).round().numpy().astype(np.uint8))
    tifffile = pytest.importorskip("tifffile")
    masks = torch.rand(1, 16, 24)
    folder, *_ = save(images, tmp_path, number_version=7, format="tiff", bit_depth="16", masks=masks, need_video=False)
    page = tifffile.TiffFile(f"{folder}/depth_v007.01001.tif").pages[0]
    data = page.asarray()
    assert data.dtype == np.uint16 and data.shape == (16, 24, 4) and page.compression in (8, 32946)   # deflate
    assert np.array_equal(data[..., :3], (images[0] * 65535).round().numpy().astype(np.uint16))
    assert np.array_equal(data[..., 3], ((1 - masks[0]) * 65535).round().numpy().astype(np.uint16))


def test_exr_is_linear_half(tmp_path, monkeypatch):
    import builtins

    import numpy as np

    images = torch.full((1, 8, 12, 3), 0.5)
    OpenEXR = pytest.importorskip("OpenEXR")

    def read(path):
        with OpenEXR.File(path) as exr:
            channel = exr.channels()["RGB"]
            return exr.header()["compression"], np.asarray(channel.pixels, np.float32)

    for number, compression in ((4, "piz"), (5, "dwaa")):
        folder, *_ = save(images, tmp_path, number_version=number, format="exr", exr_compression=compression,
                          need_video=False)
        kind, pixels = read(f"{folder}/depth_v{number:03d}.01001.exr")
        assert kind == getattr(OpenEXR, f"{compression.upper()}_COMPRESSION")
        assert abs(float(pixels.mean()) - 0.2140) < 0.002          # sRGB 0.5 in linear light

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name))
                        if name == "OpenEXR" else real_import(name, *a, **k))
    folder, *_ = save(images, tmp_path, number_version=6, format="exr", need_video=False)   # without the package
    monkeypatch.setattr(builtins, "__import__", real_import)
    kind, pixels = read(f"{folder}/depth_v006.01001.exr")
    assert kind == OpenEXR.ZIP_COMPRESSION and abs(float(pixels.mean()) - 0.2140) < 0.002


def test_async_write_returns_first_and_reports_failures_later(tmp_path, monkeypatch):
    from comfyui_tools import image_nodes

    images = torch.rand(3, 8, 10, 3)
    folder, *_ = save(images, tmp_path, async_write=True, need_video=False)
    image_nodes.wait_for_writes()
    assert sorted(p.name for p in (tmp_path / "depth_v004").iterdir()) == [
        "depth_v004.01001.png", "depth_v004.01002.png", "depth_v004.01003.png"]

    folder, _, _, video, _ = save(images, tmp_path, number_version=5, async_write=True)   # mp4 waits for the frames
    assert video and len(list((tmp_path / "depth_v005").iterdir())) == 3

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(image_nodes, "write_file", broken)
    save(images, tmp_path, number_version=6, async_write=True, need_video=False)             # returns at once
    from concurrent.futures import wait
    wait(list(image_nodes._PENDING))                                                        # they run, and fail
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="a frame written in the background failed: disk full"):
        save(images, tmp_path, number_version=7, need_video=False)                          # shows on the next run
