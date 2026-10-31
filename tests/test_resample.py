import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from comfyui_tools import image_nodes
from comfyui_tools.image_nodes import ImageResize
from comfyui_tools.resample import _apply, resize, weights


def photo(width, height, seed=0):
    rng = np.random.default_rng(seed)
    base = (rng.random((height // 8, width // 8, 3)) * 255).astype(np.uint8)
    smooth = Image.fromarray(base).resize((width, height), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.0))
    return (np.asarray(smooth, np.float32) * 0.7 + 40).round().astype(np.uint8)


@pytest.mark.parametrize("kind, pil", [("lanczos", Image.LANCZOS), ("bicubic", Image.BICUBIC),
                                       ("bilinear", Image.BILINEAR), ("box", Image.BOX)])
def test_weights_are_pillows(kind, pil):
    src = photo(288, 194)
    for w, h in ((196, 132), (577, 389), (101, 31)):
        ref = np.asarray(Image.fromarray(src).resize((w, h), pil), np.float32)
        t = torch.from_numpy(src.astype(np.float32))[None]
        xi, xw = weights(288, w, kind)
        yi, yw = weights(194, h, kind)
        mid = _apply(t, 2, xi, xw).round().clamp(0, 255)          # Pillow's 8-bit step between its passes
        emulated = _apply(mid, 1, yi, yw).round().clamp(0, 255)[0].numpy()
        assert np.abs(emulated - ref).max() <= 1                  # Pillow's fixed-point weights


def test_float_result_against_pillow_like_the_spec():
    src = photo(2881, 922)                                        # 28816x9216 -> 19584x6264, a tenth
    ref = np.asarray(Image.fromarray(src).resize((1958, 626), Image.LANCZOS), np.float32) / 255
    out = resize(torch.from_numpy(src.astype(np.float32) / 255)[None], 1958, 626, "lanczos", strip_pixels=200_000)
    diff = np.abs(out[0].clamp(0, 1).numpy() - ref)
    assert diff.mean() < 0.3 / 255 and diff.max() <= 1.5 / 255     # Pillow rounds to 8 bits twice, this is float


def test_image_resize_scale_factor_limits_and_large_frames(monkeypatch):
    spec = ImageResize.INPUT_TYPES()
    assert spec["required"]["width"][1]["max"] == 65536 and spec["required"]["width"][1]["step"] == 1
    image = torch.from_numpy(photo(200, 64).astype(np.float32) / 255)[None]
    node = ImageResize()
    args = dict(mode="scale_factor", target_side=1024, megapixels=1.0, width=1024, height=1024,
                keep_aspect=True, multiple_of=1, interpolation="lanczos", scale_factor=2.72)
    small, w, h = node.resize(image, **args)
    assert (w, h) == (544, 174) and small.shape == (1, 174, 544, 3)
    monkeypatch.setattr(ImageResize, "LARGE", 10_000)             # take the strip path
    large, w2, h2 = node.resize(image, **args)
    assert (w2, h2) == (w, h) and float((large - resize(image, 544, 174, "lanczos").clamp(0, 1)).abs().max()) == 0
    assert float(large.min()) >= 0 and float(large.max()) <= 1
    same, *_ = node.resize(image, **{**args, "scale_factor": 1.0})
    assert same is image
