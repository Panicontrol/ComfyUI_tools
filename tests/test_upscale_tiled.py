import types

import pytest
import torch

from comfyui_tools.resample import resize
from comfyui_tools.upscale_nodes import UpscaleWithModelTiled, feather, tile_starts


class Small(torch.nn.Module):
    """x2 with a 5 px receptive field: tiles overlapping by more than that give the whole-image result."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 12, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)

    def forward(self, x):
        return torch.nn.functional.pixel_shuffle(self.conv(self.conv2(x) + x), 2).sigmoid()


class Descriptor:
    """What Load Upscale Model hands over, as far as the node uses it."""

    def __init__(self, module, fail_above=None, nan_once=False):
        self.model, self.scale, self.architecture = module, 2, types.SimpleNamespace(name="Small")
        self.fail_above, self.nan_once, self.calls = fail_above, nan_once, 0

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, x):
        self.calls += 1
        if self.fail_above and x.shape[-1] > self.fail_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        out = self.model(x)
        if self.nan_once and self.calls == 1:
            out = out.clone()
            out[0, :, :2, :2] = float("nan")
        return out


def run(model, image, **kw):
    args = dict(tile=64, overlap=16, tiles_per_batch=3, precision="fp32", output_device="cpu",
                downscale_to="none", width=0, height=0, factor=2.0, interpolation="lanczos")
    args.update(kw)
    return UpscaleWithModelTiled().upscale(model, image, **args)


def whole(model, image):
    with torch.no_grad():
        return model.model(image.movedim(-1, 1)).movedim(1, -1)


def test_tiles_put_together_give_the_whole_image_result():
    image = torch.rand(2, 150, 230, 3)
    model = Descriptor(Small())
    out, info = run(model, image)
    assert out.shape == (2, 300, 460, 3) and float((out - whole(model, image)).abs().max()) < 1e-5
    assert "Small x2, tile 64 overlap 16" in info and "15 tiles a frame" in info and "460x300" in info


def test_downscale_to_matches_resizing_the_whole_result():
    image = torch.rand(1, 150, 230, 3)
    model = Descriptor(Small())
    out, info = run(model, image, downscale_to="factor", factor=1.36)
    assert out.shape == (1, 204, 313, 3)                      # 150 x 1.36, 230 x 1.36
    ref = resize(whole(model, image), 313, 204, "lanczos").clamp(0, 1)
    assert float((out - ref).abs().max()) < 1e-4 and "(lanczos from 460x300)" in info
    out, _ = run(model, image, downscale_to="width_height", width=300, height=100, interpolation="area")
    assert float((out - resize(whole(model, image), 300, 100, "box").clamp(0, 1)).abs().max()) < 1e-4


def test_alpha_out_of_memory_and_nan_tiles():
    image = torch.rand(1, 100, 140, 4)
    out, info = run(Descriptor(Small(), fail_above=128), image, tile=256)   # 256 px runs out: 128 does it
    assert out.shape == (1, 200, 280, 4) and "tile 128" in info
    assert float((out[..., 3] - resize(image[..., 3:], 280, 200, "bilinear")[..., 0].clamp(0, 1)).abs().max()) < 1e-5
    model = Descriptor(Small(), nan_once=True)
    out, info = run(model, image[..., :3])
    assert torch.isfinite(out).all() and float((out - whole(model, image[..., :3])).abs().max()) < 1e-5
    assert "1 tile(s) again in fp32" in info


def test_tile_layout_and_feather():
    assert tile_starts(230, 64, 16) == [0, 48, 96, 144, 166] and tile_starts(50, 64, 16) == [0]
    ramps = feather([(0, 128), (96, 224), (192, 300)])
    total = torch.zeros(300)
    for (a, b), w in zip([(0, 128), (96, 224), (192, 300)], ramps):
        total[a:b] += w
    assert torch.allclose(total, torch.ones(300))              # a partition of one


def test_with_a_spandrel_model():
    spandrel = pytest.importorskip("spandrel")
    from spandrel.architectures.SPAN import SPAN

    net = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=16, upscale=4)
    model = spandrel.ModelLoader().load_from_state_dict(net.state_dict()).eval()
    image = torch.rand(1, 96, 120, 3)
    out, info = run(model, image, tile=64, overlap=24)
    assert out.shape == (1, 384, 480, 3) and torch.isfinite(out).all() and info.startswith("SPAN x4")
