import safetensors.torch
import torch

from comfyui_tools import latent_nodes
from comfyui_tools.latent_nodes import LoadLatentFrom, SaveLatentTo


class FakeNested:
    """Stand-in for comfy.nested_tensor.NestedTensor."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True


def test_round_trip_to_a_chosen_folder(tmp_path):
    latent = {
        "samples": torch.randn(1, 16, 4, 8, 8, dtype=torch.bfloat16),
        "noise_mask": torch.rand(1, 1, 4, 8, 8),
        "batch_index": [0],
    }
    folder = tmp_path / "shots" / "sh010"  # created on save
    _, first = SaveLatentTo().save(latent, str(folder), "h3")["result"]
    _, second = SaveLatentTo().save(latent, str(folder), "h3")["result"]
    assert first.endswith("h3_00001.latent") and second.endswith("h3_00002.latent")

    loaded = LoadLatentFrom().load(first)[0]
    assert torch.equal(loaded["samples"], latent["samples"])  # dtype kept
    assert torch.equal(loaded["noise_mask"], latent["noise_mask"])
    assert loaded["batch_index"] == [0]


def test_minimax_av_nested_latent_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(latent_nodes, "NestedTensor", FakeNested)
    video, audio = torch.randn(1, 16, 32, 8, 8), torch.randn(1, 32, 2, 50)
    latent = {
        "samples": FakeNested([video, audio]),
        "noise_mask": FakeNested([torch.ones(32, 8, 8), torch.zeros(32)]),
    }
    _, path = SaveLatentTo().save(latent, str(tmp_path), "av")["result"]

    loaded = LoadLatentFrom().load(path)[0]
    assert isinstance(loaded["samples"], FakeNested)
    assert torch.equal(loaded["samples"].tensors[0], video)
    assert torch.equal(loaded["samples"].tensors[1], audio)
    assert loaded["noise_mask"].tensors[1].shape == (32,)


def test_reads_comfyui_own_latent_files(tmp_path):
    tensor = torch.randn(1, 4, 8, 8)
    path = tmp_path / "ComfyUI_00001_.latent"
    safetensors.torch.save_file(
        {"latent_tensor": tensor, "latent_format_version_0": torch.tensor([])}, str(path)
    )
    assert torch.equal(LoadLatentFrom().load(str(path))[0]["samples"], tensor)


def test_saved_file_follows_the_umask(tmp_path):
    import os
    import stat

    old = os.umask(0o022)
    try:
        _, path = SaveLatentTo().save({"samples": torch.zeros(1, 4, 2, 2)}, str(tmp_path), "p")["result"]
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o644


def test_encode_still_swaps_only_the_video_part():
    import pytest

    from comfyui_tools.latent_nodes import EncodeStillH3

    class StillVAE:  # ComfyUI's H3 VAE on one image: [1, 24, 1, H/16, W/16]
        def encode(self, pixels):
            return torch.ones(1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)

    audio = torch.randn(1, 32, 2, 2)
    still = {"samples": FakeNested([torch.zeros(1, 24, 1, 4, 6), audio]), "extra": 1}
    image = torch.rand(1, 50, 70, 3)  # any size: resized to the latent's 64x96 canvas

    out = EncodeStillH3().encode(StillVAE(), image, still)[0]
    video_out, audio_out = out["samples"].tensors
    assert video_out.shape == (1, 24, 1, 4, 6) and video_out.min() == 1.0
    assert audio_out is audio and out["extra"] == 1

    with pytest.raises(ValueError, match="video \\+ audio"):
        EncodeStillH3().encode(StillVAE(), image, {"samples": torch.zeros(1, 24, 1, 4, 6)})
    with pytest.raises(ValueError, match="one still at a time"):
        EncodeStillH3().encode(StillVAE(), torch.rand(2, 64, 96, 3), still)
