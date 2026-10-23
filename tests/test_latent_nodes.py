import pytest
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


def test_h3_mask_merge_maps_frames_to_tokens():
    from comfyui_tools.latent_nodes import H3LatentMaskMerge

    # 6 tokens = 1 + 4 + 4 + 4 + 4 + 1 = 18 frames; latent 4x6 = 64x96 px
    va, vb = torch.zeros(1, 24, 6, 4, 6), torch.ones(1, 24, 6, 4, 6)
    a = {"samples": FakeNested([va, torch.full((1, 32, 2, 30), 7.0)])}
    b = {"samples": FakeNested([vb, torch.zeros(1, 32, 2, 30)])}
    mask = torch.ones(18, 64, 96)
    mask[7, :16, :16] = 0  # frame 7 lives in token 2 (frames 5-8)

    out = H3LatentMaskMerge().merge(a, b, mask, set_noise_mask=True, feather=0)[0]
    video, audio = out["samples"].tensors
    assert [float(video[0, 0, k, 0, 0]) for k in range(6)] == [1, 1, 0, 1, 1, 1]
    assert float(video[0, 0, 2, 3, 5]) == 1.0 and torch.equal(audio, a["samples"].tensors[1])
    assert out["noise_mask"].tensors[0].shape == (1, 1, 6, 4, 6)


def test_h3_mask_merge_rejects_different_sizes():
    from comfyui_tools.latent_nodes import H3LatentMaskMerge

    a = {"samples": FakeNested([torch.zeros(1, 24, 6, 4, 6), torch.zeros(1, 32, 2, 30)])}
    b = {"samples": FakeNested([torch.zeros(1, 24, 6, 8, 12), torch.zeros(1, 32, 2, 30)])}
    with pytest.raises(ValueError, match="video parts differ"):
        H3LatentMaskMerge().merge(a, b, torch.ones(64, 96))


def test_h3_mask_merge_feather_and_grow_soften_the_edge():
    from comfyui_tools.latent_nodes import mask_to_h3_latent

    mask = torch.zeros(1, 64, 96)
    mask[:, :, 48:] = 1.0  # hard edge on a latent cell border (column 3)
    hard = mask_to_h3_latent(mask, 1, 4, 6)[0, 0, 0, 0]
    soft = mask_to_h3_latent(mask, 1, 4, 6, feather=16)[0, 0, 0, 0]
    grown = mask_to_h3_latent(mask, 1, 4, 6, grow=16)[0, 0, 0, 0]
    assert hard.tolist() == [0, 0, 0, 1, 1, 1]
    assert 0 < soft[2] < 0.5 < soft[3] < 1  # the edge now spans the cells around it
    assert grown.tolist() == [0, 0, 1, 1, 1, 1]  # 16 px more white = one cell


def h3_latent(tokens=12, rows=8, cols=12, value=0.0):
    frames = latent_nodes.frames_for_tokens(tokens)
    return {"samples": FakeNested([torch.full((1, 24, tokens, rows, cols), value),
                                   torch.arange(round(frames * 5 / 3), dtype=torch.float32).expand(1, 32, 2, -1)])}


def test_h3_crop_snaps_to_the_h3_grid_and_pastes_back():
    from comfyui_tools.latent_nodes import H3LatentCrop, H3LatentPaste

    base = h3_latent()  # 12 tokens = 39 frames, 192x128 px
    piece, crop, mask, info, _ = H3LatentCrop().crop(base, 40, 20, 50, 40, 20, 10)
    # box widened to the 32 px grid, frames snapped to start 17 and length 22 (17 + 5)
    assert (crop["x"], crop["y"], crop["width"], crop["height"]) == (32, 0, 64, 64)
    assert (crop["start_frame"], crop["frames"], crop["token0"], crop["tokens"]) == (17, 22, 5, 7)
    video, audio = piece["samples"].tensors
    assert video.shape == (1, 24, 7, 4, 4) and float(audio[0, 0, 0, 0]) == round(17 * 5 / 3)
    assert mask.shape == (1, 64, 64)

    sampled = {"samples": FakeNested([torch.ones_like(video), audio])}
    out = H3LatentPaste().paste(base, sampled, crop, feather=0)[0]["samples"].tensors[0]
    assert float(out[0, 0, 6, 1, 3]) == 1.0                    # inside the box
    assert float(out[0, 0, 6, 1, 1]) == 0.0 and float(out[0, 0, 2, 1, 3]) == 0.0  # outside, before

    soft = H3LatentPaste().paste(base, sampled, crop, feather=32)[0]["samples"].tensors[0]
    assert float(soft[0, 0, 6, 0, 2]) < 0.5 < float(soft[0, 0, 6, 1, 3])  # fades in from the box edge
    assert float(soft[0, 0, 6, 0, 1]) == 0.0                   # nothing outside the box


def test_h3_crop_around_a_mask():
    from comfyui_tools.latent_nodes import H3LatentCrop

    mask = torch.zeros(1, 128, 192)
    mask[:, 70:90, 100:120] = 1.0
    _, crop, cut, _, _ = H3LatentCrop().crop(h3_latent(), 0, 0, 0, 0, 0, 0, mask=mask, padding=8)
    assert (crop["x"], crop["y"], crop["width"], crop["height"]) == (64, 32, 64, 96)
    assert crop["frames"] == 39 and float(cut.max()) == 1.0


def test_h3_diff_mask_finds_the_edit_not_the_drift():
    from comfyui_tools.latent_nodes import H3LatentDiffMask

    g = torch.Generator().manual_seed(0)
    a = torch.randn(1, 24, 12, 8, 12, generator=g)
    b = a + 0.3 + 0.1 * torch.randn(a.shape, generator=g)  # colour drift + regeneration noise everywhere
    b[:, :, 5:12, 2:6, 6:10] = torch.randn(1, 24, 7, 4, 4, generator=g)  # the actual edit
    audio = torch.zeros(1, 32, 2, 65)
    la, lb = {"samples": FakeNested([a, audio])}, {"samples": FakeNested([b, audio])}

    mask, heat, info = H3LatentDiffMask().diff(la, lb, temporal_smooth=0, grow=0, feather=0)
    assert mask.shape == (39, 128, 192) and heat.shape == (39, 32, 48)  # 12 tokens = 39 frames
    assert float(mask[30, 64, 128]) == 1.0      # inside the edit (token 9, cell 4,8)
    assert float(mask[30, 16, 16]) == 0.0       # drifted but unedited
    assert float(mask[10, 64, 128]) == 0.0      # same place before the edit starts (token 3)
    assert "auto" in info


def test_h3_decode_frames_picks_frames_from_partial_decodes():
    from comfyui_tools.latent_nodes import H3DecodeFrames, frames_for_tokens, parse_frame_list

    assert parse_frame_list("0, 5-7, 0-40:20, -1", 124) == [0, 5, 6, 7, 20, 40, 123]

    class TokenVAE:  # each frame shows the absolute index of its token (channel 0 holds it)
        calls = []

        def decode(self, z):
            self.calls.append(z.shape[2])
            ids = z[0, 0, :, 0, 0]
            frames = [ids[k] for k in range(z.shape[2]) for _ in range(1 if k % 5 == 0 else 4)]
            return torch.stack(frames).view(1, -1, 1, 1, 1).expand(1, -1, 2, 2, 3)

    tokens = 37  # 124 frames
    z = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1, 1).expand(1, 24, tokens, 1, 1)
    vae = TokenVAE()
    images, frames = H3DecodeFrames().decode(vae, {"samples": z}, "100, 3, -1", exact=True)
    owner = [k for k in range(tokens) for _ in range(frames_for_tokens(k + 1) - frames_for_tokens(k))]
    assert frames == "3, 100, 123"
    assert [int(images[i, 0, 0, 0]) for i in range(3)] == [owner[3], owner[100], owner[123]]
    assert sum(vae.calls) < tokens  # never the whole latent


def test_latent_info_reads_h3_and_flags_problems():
    from comfyui_tools.latent_nodes import LatentInfo

    latent = h3_latent(tokens=37, rows=68, cols=120)  # 124 frames, 1920x1088
    out = LatentInfo().inspect(latent)
    text, frames, width, height = out["result"][1:]
    assert (frames, width, height) == (124, 1920, 1088) and out["ui"]["text"] == [text]
    assert "124 frames = 37 tokens, 5.17 s at 24 fps, on the 17k+5 grid" in text
    assert "207 latent steps = 5.17 s" in text and "all zeros (empty latent)" in text

    odd = {"samples": FakeNested([torch.full((1, 24, 3, 5, 6), float("nan")), torch.zeros(1, 32, 2, 10)])}
    text = LatentInfo().inspect(odd)["result"][1]
    assert "NOT on the 17k+5 grid" in text and "not multiples of 32 px" in text and "all NaN/Inf" in text


def test_h3_crop_extended_mask_options():
    from comfyui_tools.latent_nodes import H3LatentCrop

    latent = h3_latent(tokens=37, rows=34, cols=60)  # 124 frames, 960x544 px
    mask = torch.zeros(124, 544, 960)
    mask[40:70, 200:296, 400:464] = 1.0      # the object: 64x96 px, frames 40-69
    mask[:, 10:14, 10:14] = 1.0              # a 4 px speck in every frame

    _, crop, cut, info, preview = H3LatentCrop().crop(
        latent, 0, 0, 0, 0, 0, 0, mask=mask, padding=0, despeckle=8, padding_percent=50,
        time_from_mask=True, time_padding=4, aspect_ratio="1:1", min_size=256)
    # frames: white 40-69, +-4 -> 36-73, snapped to start 34, 17k+5 long
    assert (crop["start_frame"], crop["frames"]) == (34, 56) and cut.shape[0] == 56
    # box: object + 50% each side = 128x192, square 192, min 256 -> 256x256 around (432, 248), on the 32 px grid
    assert (crop["width"], crop["height"]) == (288, 288) and (crop["x"], crop["y"]) == (288, 96)
    assert "short side 288 px" in info and "white in frames 40-69" in info
    assert preview.shape == (1, 544, 960, 3)

    # without time_from_mask the box only looks at the frames being cropped: frames 0-21 have just the speck
    with pytest.raises(ValueError, match="empty in frames 0-21"):
        H3LatentCrop().crop(latent, 0, 0, 0, 0, 0, 22, mask=mask, despeckle=8)


def test_h3_latent_resize_downscales_video_only():
    from comfyui_tools.latent_nodes import H3LatentResize

    latent = h3_latent(tokens=37, rows=96, cols=168)          # 2688x1536
    latent["samples"].tensors[0].fill_(2.0)
    latent["noise_mask"] = FakeNested([torch.ones(1, 1, 37, 96, 168), torch.zeros(1, 32, 2, 207)])
    out, w, h = H3LatentResize().resize(latent, scale=0.5)
    video, audio = out["samples"].tensors
    assert (w, h) == (1344, 768) and video.shape == (1, 24, 37, 48, 84)
    assert torch.allclose(video, torch.full_like(video, 2.0))  # area keeps flat values
    assert torch.equal(audio, latent["samples"].tensors[1])
    assert out["noise_mask"].tensors[0].shape == (1, 1, 37, 48, 84)

    _, w, h = H3LatentResize().resize(latent, width=1000)      # height from the aspect, both on the 32 px grid
    assert (w, h) == (992, 576)
