"""Shared helpers used by the node packs in this repository.

ComfyUI tensor conventions used throughout:

* ``IMAGE`` -- ``torch.Tensor`` of shape ``[B, H, W, C]``, float32 in ``[0, 1]``
* ``MASK``  -- ``torch.Tensor`` of shape ``[B, H, W]``, float32 in ``[0, 1]``
"""

import torch
import torch.nn.functional as F

try:  # available when running inside ComfyUI
    from comfy.utils import common_upscale as _comfy_upscale
except Exception:  # pragma: no cover - exercised only outside ComfyUI
    _comfy_upscale = None

CATEGORY = "tools"

INTERPOLATION_MODES = ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]

# Fallbacks for modes torch cannot do natively.
_TORCH_MODES = {
    "lanczos": "bicubic",
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "area": "area",
    "nearest-exact": "nearest-exact",
}


class AnyType(str):
    """A socket type that ComfyUI accepts as a match for every other type."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


def round_to_multiple(value, multiple, minimum=1):
    """Round ``value`` to the nearest multiple of ``multiple``."""
    if multiple <= 1:
        return max(minimum, int(round(value)))
    rounded = int(round(value / multiple)) * multiple
    return max(minimum if minimum % multiple else multiple, rounded)


def image_to_bchw(image):
    return image.movedim(-1, 1)


def bchw_to_image(tensor):
    return tensor.movedim(1, -1)


def resize_image(image, width, height, interpolation="lanczos"):
    """Resize a ``[B, H, W, C]`` image tensor to ``width`` x ``height``."""
    if image.shape[2] == width and image.shape[1] == height:
        return image

    samples = image_to_bchw(image)
    if _comfy_upscale is not None:
        samples = _comfy_upscale(samples, width, height, interpolation, "disabled")
    else:
        mode = _TORCH_MODES.get(interpolation, "bicubic")
        kwargs = {"antialias": True} if mode in ("bicubic", "bilinear") else {}
        samples = F.interpolate(samples, size=(height, width), mode=mode, **kwargs)
    return bchw_to_image(samples).clamp(0.0, 1.0)


def resize_mask(mask, width, height, interpolation="bilinear"):
    """Resize a ``[B, H, W]`` mask tensor to ``width`` x ``height``."""
    if mask.shape[2] == width and mask.shape[1] == height:
        return mask
    samples = mask.unsqueeze(1)
    mode = _TORCH_MODES.get(interpolation, "bilinear")
    kwargs = {"antialias": True} if mode in ("bicubic", "bilinear") else {}
    samples = F.interpolate(samples, size=(height, width), mode=mode, **kwargs)
    return samples.squeeze(1).clamp(0.0, 1.0)


def gaussian_blur(tensor, radius):
    """Blur a ``[B, 1, H, W]`` tensor with a separable Gaussian kernel."""
    if radius <= 0:
        return tensor

    sigma = max(radius / 2.0, 1e-6)
    size = int(radius) * 2 + 1
    coords = torch.arange(size, dtype=tensor.dtype, device=tensor.device) - radius
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    channels = tensor.shape[1]
    horizontal = kernel.view(1, 1, 1, size).expand(channels, 1, 1, size)
    vertical = kernel.view(1, 1, size, 1).expand(channels, 1, size, 1)

    padded = F.pad(tensor, (int(radius), int(radius), 0, 0), mode="reflect")
    blurred = F.conv2d(padded, horizontal, groups=channels)
    padded = F.pad(blurred, (0, 0, int(radius), int(radius)), mode="reflect")
    return F.conv2d(padded, vertical, groups=channels)


def hex_to_rgb(color):
    """Parse ``#rrggbb``/``rrggbb`` or ``r,g,b`` (0-255) into floats in ``[0, 1]``."""
    text = str(color).strip()
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"expected 3 comma separated channels, got: {color!r}")
        values = [float(p) for p in parts]
    else:
        text = text.lstrip("#")
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        if len(text) != 6:
            raise ValueError(f"not a hex color: {color!r}")
        values = [int(text[i:i + 2], 16) for i in (0, 2, 4)]
    return tuple(min(max(v, 0.0), 255.0) / 255.0 for v in values)
