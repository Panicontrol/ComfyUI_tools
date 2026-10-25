import math
import os
import zlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from comfyui_tools.degrid_nodes import H3DeGrid, grid_ratio, measure


def smooth(x, s):
    r = int(math.ceil(3 * s))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / s) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.transpose(2, 3), groups=x.shape[1])


def plate(h=512, w=768, seed=0):
    """Smooth content with an edge, a textured patch and grain, [1, 3, H, W]."""
    g = torch.Generator().manual_seed(seed)
    x = 0.45 + 0.15 * smooth(torch.randn(1, 3, h, w, generator=g), 30) / 0.02
    x = 0.45 + (x - 0.45).clamp(-0.2, 0.2)
    x[..., h // 2:, :] += 0.15
    x[..., 20:80, 20:120] += 0.03 * torch.randn(1, 3, 60, 100, generator=g)
    return (x + 0.45 / 255 * torch.randn(1, 3, h, w, generator=g)).clamp(0, 1)


def h3_like(x, frame=0):
    """x with an 8 px block grid (plateaus and per-block offsets) and a 2 px parity."""
    half = F.avg_pool2d(x, 2)
    out = x + 0.48 * (F.interpolate(half, scale_factor=2, mode="nearest")
                      - F.interpolate(half, scale_factor=2, mode="bilinear", align_corners=False))
    low = smooth(out, 2.0)
    out = out + 0.4 * (F.interpolate(F.avg_pool2d(low, 8), scale_factor=8, mode="nearest") - low)
    g = torch.Generator().manual_seed(100 + frame)
    off = torch.randn(1, 3, x.shape[2] // 8, x.shape[3] // 8, generator=g) * 0.3 / 255
    return (out + F.interpolate(off, scale_factor=8, mode="nearest")).clamp(0, 1)


def degrid(images, **kw):
    out = H3DeGrid().degrid(images, **kw)
    assert out["ui"]["text"] == [out["result"][2]]   # the info shows on the node too
    return out["result"]


def ratios(images):
    _, px, py = measure(images.permute(0, 3, 1, 2))
    r = lambda p, period: float(grid_ratio(p, period)[:, 0].mean())
    return r(px, 8.0), r(py, 8.0), r(px, 2.0), r(py, 2.0)


def as_images(x):
    return x.permute(0, 2, 3, 1).contiguous()


def test_degrid_removes_grid_and_parity_keeps_float_and_alpha():
    base = plate()
    images = as_images(torch.cat([h3_like(base, i) for i in range(2)]))
    before = ratios(images)
    assert min(before[:2]) > 1.3 and min(before[2:]) > 1.15

    out, removed, info = degrid(images)
    after = ratios(out)
    assert max(after) < 1.10, after   # noisier than on full-size frames
    assert out.shape == images.shape and removed.shape == images.shape
    assert not torch.allclose(out * 255, (out * 255).round())               # float, not 8-bit steps
    assert float((out - images).abs().max()) <= 5.6 / 255                    # bounded corrections
    assert "grid x/y (R)" in info and "0 of 2 frames clean" in info

    rgba = torch.cat([images, torch.rand(2, *images.shape[1:3], 1)], -1)
    out4, removed4, _ = degrid(rgba)
    assert out4.shape == rgba.shape and torch.equal(out4[..., 3], rgba[..., 3])
    assert torch.allclose(out4[..., :3], out, atol=1e-6) and removed4.shape[-1] == 3


def test_degrid_leaves_clean_and_resized_frames_alone():
    clean = as_images(plate())
    out, removed, info = degrid(clean)
    assert torch.equal(out, clean) and "1 of 1 frames clean" in info
    assert torch.allclose(removed, torch.full_like(removed, 0.5))

    # a 4/3 resize moves the grid to 10.67 px: not this node's 8 px grid
    big = F.interpolate(h3_like(plate(h=384, w=576)), scale_factor=4 / 3, mode="bicubic", align_corners=False)
    big = as_images(big.clamp(0, 1))
    out, _, info = degrid(big)
    assert torch.equal(out, big) and "another period" in info


def test_degrid_finds_period_and_phase_once():
    shifted = F.pad(h3_like(plate()), (5, 0, 3, 0), mode="replicate")   # boundaries at 8k+5 / 8k+3
    images = as_images(shifted)
    out, _, info = degrid(images, period=0, phase_x=-1, phase_y=-1, h3_sizes_only=False)   # 515 x 773
    assert "period 8, phase x 5 / y 3" in info and "period 8 found" in info
    _, px, py = measure(out.permute(0, 3, 1, 2))
    assert float(grid_ratio(px, 8.0, 5)[0, 0]) < 1.10 and float(grid_ratio(py, 8.0, 3)[0, 0]) < 1.10


def test_degrid_passes_non_h3_sizes_through():
    images = as_images(h3_like(plate(h=512, w=768)))[:, :504, :760]     # like fal 4K's 2160: not on 32
    out, removed, info = degrid(images)
    assert torch.equal(out, images) and "760x504 is not an H3 size" in info
    assert torch.allclose(removed, torch.full_like(removed, 0.5))
    out, _, _ = degrid(images, h3_sizes_only=False)
    assert not torch.equal(out, images)


def sill(h=512, w=768):
    """Horizontal edges off the 8 px grid (a windowsill and its shadows) on a falloff, with grain."""
    g = torch.Generator().manual_seed(3)
    yy = torch.arange(h, dtype=torch.float32)[:, None].expand(h, w)
    img = 0.35 + 0.15 * yy / h
    y = 40.0
    while y < h - 40:
        img = img + float(torch.empty(1).uniform_(-0.06, 0.06, generator=g)) * torch.sigmoid(
            (yy - y) / float(torch.empty(1).uniform_(0.3, 2.5, generator=g)))
        y += float(torch.empty(1).uniform_(11, 45, generator=g))
    img = img[None, None].repeat(1, 3, 1, 1)
    return (img + 0.45 / 255 * torch.randn(1, 3, h, w, generator=g)).clamp(0, 1)


def test_plateau_does_not_shift_edges_through_blocks():
    images = as_images(torch.cat([h3_like(sill(), i) for i in range(2)]))
    off, _, _ = degrid(images)
    on, _, _ = degrid(images, plateau="on")
    assert ratios(on)[1] <= ratios(off)[1] + 0.01          # y: the edges' direction
    flicker = lambda t: float((t[1] - t[0]).abs().mean())
    assert flicker(on) <= flicker(off) * 1.02


def test_edge_weight_choices():
    images = as_images(h3_like(plate()))
    grain, _, _ = degrid(images)                        # the default
    linear, _, _ = degrid(images, edge_weight="linear")
    assert not torch.equal(grain, linear)
    # which one is better is for real frames to say (above grain on shot_001); both remove the grid here
    assert max(ratios(grain)[:2]) < 1.10 and max(ratios(linear)[:2]) < 1.10


def test_degrid_mask_strength_and_plateau():
    images = as_images(h3_like(plate()))
    mask = torch.zeros(1, images.shape[1], images.shape[2])
    out, _, _ = degrid(images, mask=mask)
    assert torch.equal(out, images)
    mask[:, :, : images.shape[2] // 2] = 1
    out, _, _ = degrid(images, mask=mask)
    w = images.shape[2] // 2
    assert torch.equal(out[:, :, w:], images[:, :, w:]) and not torch.equal(out[:, :, :w], images[:, :, :w])

    full, _, _ = degrid(images)
    half, _, _ = degrid(images, strength=0.5)
    assert 0.3 < float((half - images).abs().mean() / (full - images).abs().mean()) < 0.7
    flat, _, info = degrid(images, plateau="on")
    assert max(ratios(flat)[:2]) < 1.10 and not torch.equal(flat, full)


def read_png16(path):
    """Decode the PNG files png16_bytes writes (Sub filter rows) to uint16 [H, W, C]."""
    data = open(path, "rb").read()
    pos, idat, header = 8, b"", None
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            header = body
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    w, h = int.from_bytes(header[:4], "big"), int.from_bytes(header[4:8], "big")
    assert header[8] == 16
    c = {2: 3, 6: 4}[header[9]]
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * c * 2)
    assert (rows[:, 0] == 1).all()
    lanes = rows[:, 1:].reshape(h, w, c * 2).astype(np.int64)
    raw = (np.cumsum(lanes, axis=1) % 256).astype(np.uint8).reshape(h, w * c * 2)
    return raw.view(">u2").reshape(h, w, c).astype(np.uint16)


def test_save_16_bit_png_round_trip(tmp_path):
    from PIL import Image

    from comfyui_tools.image_nodes import SaveImageSequence

    images = torch.rand(2, 9, 13, 3)
    masks = torch.rand(2, 9, 13)
    SaveImageSequence().save(images, str(tmp_path), "deg", 1, 1001, 4, False, masks=masks, bit_depth="16")
    path = os.path.join(str(tmp_path), "deg_v001", "deg_v001.1002.png")
    got = read_png16(path)
    assert np.array_equal(got[..., :3], (images[1] * 65535).round().numpy().astype(np.uint16))
    assert np.array_equal(got[..., 3], ((1 - masks[1]) * 65535).round().numpy().astype(np.uint16))
    with Image.open(path) as opened:     # a valid PNG for any reader
        assert opened.size == (13, 9)

    SaveImageSequence().save(images, str(tmp_path), "deg8", 1, 1001, 4, False)
    with Image.open(os.path.join(str(tmp_path), "deg8_v001", "deg8_v001.1001.png")) as opened:
        assert opened.mode == "RGB"      # 8 bits stays the default


def test_load_reads_16_bit_png(tmp_path, monkeypatch):
    from comfyui_tools import image_nodes

    path = str(tmp_path / "f.png")
    values = (np.random.rand(5, 7, 3) * 65535).astype(np.uint16)
    with open(path, "wb") as handle:
        handle.write(image_nodes.png16_bytes(values))
    assert image_nodes.png_bit_depth(path) == 16

    class FakeCv2:   # stands in for OpenCV: BGR order, uint16
        IMREAD_UNCHANGED = -1

        @staticmethod
        def imread(p, flag):
            return read_png16(p)[..., ::-1].copy()

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCv2)
    frame, mask = image_nodes.load_frame(path)
    assert torch.allclose(frame[0], torch.from_numpy(values.astype(np.float32) / 65535), atol=1e-6)
    assert float(mask.abs().max()) == 0.0

    monkeypatch.setitem(__import__("sys").modules, "cv2", None)   # no OpenCV: Pillow, 8-bit precision
    frame, _ = image_nodes.load_frame(path)
    assert torch.allclose(frame[0], torch.from_numpy(values.astype(np.float32) / 65535), atol=1.5 / 255)


def test_save_dither_breaks_up_8_bit_steps(tmp_path):
    from PIL import Image

    from comfyui_tools.image_nodes import SaveImageSequence

    ramp = torch.linspace(0.5, 0.5 + 2 / 255, 64).view(1, 1, 64, 1).expand(1, 16, 64, 3)  # 2 steps in 64 px
    for dither in (False, True):
        SaveImageSequence().save(ramp, str(tmp_path), f"d{int(dither)}", 1, 1, 4, False, dither=dither)
    plain = np.array(Image.open(os.path.join(str(tmp_path), "d0_v001", "d0_v001.0001.png")), np.float32)
    dithered = np.array(Image.open(os.path.join(str(tmp_path), "d1_v001", "d1_v001.0001.png")), np.float32)
    assert len(np.unique(plain[0, :, 0])) == 3 and (np.diff(plain[0, :, 0]) != 0).sum() == 2   # two hard steps
    assert (np.diff(dithered[0, :, 0]) != 0).sum() > 10                                      # broken up
    target = ramp[0].numpy() * 255
    assert abs(float(dithered.mean() - target.mean())) < 0.05 and np.abs(dithered - target).max() <= 1.5
    SaveImageSequence().save(ramp, str(tmp_path), "d2", 1, 1, 4, False, dither=True)
    again = np.array(Image.open(os.path.join(str(tmp_path), "d2_v001", "d2_v001.0001.png")), np.float32)
    assert np.array_equal(again, dithered)                                                   # same on a rerun
