"""Separable resampling with Pillow's filters, in float, a strip of rows at a time.

The weights are Pillow's own (ImagingResample: the filter widened by the scale
when shrinking, every output pixel's taps normalised), so a lanczos result
matches Pillow's to within its 8-bit rounding -- without Pillow's 8-bit
detour, any size limit, or two full frames in memory: the output is built a
strip of rows at a time from the few input rows each strip needs.
"""

import math

import torch

FILTERS = {"box": 0.5, "bilinear": 1.0, "bicubic": 2.0, "lanczos": 3.0}  # support
STRIP_PIXELS = 16_000_000  # output pixels per strip


def _kernel(kind, x):
    if kind == "box":  # Pillow: -0.5 < x <= 0.5
        return ((x > -0.5) & (x <= 0.5)).to(x.dtype)
    x = x.abs()
    if kind == "bilinear":
        return (1 - x).clamp(min=0)
    if kind == "bicubic":  # a = -0.5, as Pillow
        a = -0.5
        near = ((a + 2) * x - (a + 3)) * x * x + 1
        far = (((x - 5) * x + 8) * x - 4) * a
        return torch.where(x < 1, near, torch.where(x < 2, far, torch.zeros_like(x)))

    def sinc(v):
        return torch.where(v == 0, torch.ones_like(v), torch.sin(math.pi * v) / (math.pi * v))

    return torch.where(x < 3, sinc(x) * sinc(x / 3), torch.zeros_like(x))


def weights(in_size, out_size, kind="lanczos"):
    """(index [out, taps], weight [out, taps]) taking ``in_size`` samples to ``out_size``."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = FILTERS[kind] * filterscale
    centre = (torch.arange(out_size, dtype=torch.float64) + 0.5) * scale
    low = torch.clamp((centre - support + 0.5).trunc(), min=0).long()
    high = torch.clamp((centre + support + 0.5).trunc(), max=in_size).long()
    taps = int((high - low).max())
    index = low[:, None] + torch.arange(taps)[None, :]
    valid = index < high[:, None]
    w = _kernel(kind, (index.double() - centre[:, None] + 0.5) / filterscale) * valid
    w = w / w.sum(1, keepdim=True).clamp(min=1e-12)
    return index.clamp(max=in_size - 1), w.float()


def _apply(t, dim, index, w):
    out = None
    for tap in range(index.shape[1]):
        shape = [1] * t.dim()
        shape[dim] = index.shape[0]
        part = t.index_select(dim, index[:, tap]) * w[:, tap].view(shape)
        out = part if out is None else out + part
    return out


def resize(image, width, height, kind="lanczos", device=None, strip_pixels=STRIP_PIXELS):
    """``image`` [B, H, W, C] resized to ``width`` x ``height`` with Pillow's ``kind`` filter, in float.

    The output is made a strip of rows at a time (on ``device`` if given, e.g. the GPU), from just the
    input rows that strip needs, and gathered on the image's own device.
    """
    batch, in_h, in_w, channels = image.shape
    device = device or image.device
    xi, xw = weights(in_w, width, kind)
    yi, yw = weights(in_h, height, kind)
    xi, xw = xi.to(device), xw.to(device)
    out = torch.empty((batch, height, width, channels), dtype=torch.float32, device=image.device)
    rows = max(1, int(strip_pixels // max(width * channels * batch, 1)))
    for r0 in range(0, height, rows):
        r1 = min(height, r0 + rows)
        lo, hi = int(yi[r0:r1].min()), int(yi[r0:r1].max()) + 1
        slab = image[:, lo:hi].to(device, torch.float32)
        slab = _apply(slab, 2, xi, xw)                                   # across: [B, rows in, width, C]
        out[:, r0:r1] = _apply(slab, 1, (yi[r0:r1] - lo).to(device), yw[r0:r1].to(device)).to(image.device)
    return out
