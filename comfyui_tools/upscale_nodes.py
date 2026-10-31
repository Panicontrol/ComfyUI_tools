"""Upscale With Model (tiled): an upscale model run tile by tile, in half precision, a few tiles at a time.

ComfyUI's own node runs fp32, 512 px tiles one at a time. Here the tile size,
overlap and tiles per batch are inputs, the model runs under autocast (fp16 or
bf16; a tile that comes out NaN is run again in fp32), tiles are blended with
a linear feather over the overlap, and the result is put together on the CPU
in float32 -- optionally already resized to the size wanted, tile by tile, so
a 4x result never exists at full size (28816 x 9216 x 3 floats is 3.2 GB).
"""

import logging
import math
import time

import torch

from .resample import weights
from .utils import CATEGORY

PRECISIONS = ["fp16", "bf16", "fp32"]
DOWNSCALE = ["none", "width_height", "factor"]
INTERPOLATIONS = {"lanczos": "lanczos", "bicubic": "bicubic", "area": "box"}


def tile_starts(size, tile, overlap):
    """Starts of tiles of ``tile`` px covering ``size`` with at least ``overlap``; the last one ends at the edge."""
    if size <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, size - tile, stride)) + [size - tile]
    return sorted(set(starts))


def target_plan(out_size, target, kind):
    """Per output position: which model-output samples it reads (index, weight); identity when no resize."""
    if target == out_size or kind is None:
        return None
    return weights(out_size, target, kind)


def valid_range(plan, lo, hi, size):
    """Target positions whose taps all fall inside model-output [lo, hi) -- computable from one tile."""
    if plan is None:
        return lo, hi
    index, w = plan
    used = w != 0
    first = torch.where(used, index, torch.full_like(index, size)).min(1).values
    last = torch.where(used, index, torch.full_like(index, -1)).max(1).values
    inside = (first >= lo) & (last < hi)
    picks = inside.nonzero().flatten()
    if picks.numel() == 0:
        return 0, 0
    return int(picks[0]), int(picks[-1]) + 1


def feather(ranges):
    """1-D blend weights per tile over its [a, b) target range: a linear ramp across each overlap."""
    ramps = []
    for i, (a, b) in enumerate(ranges):
        w = torch.ones(max(b - a, 0))
        if i > 0 and ranges[i - 1][1] > a:                      # overlap with the previous tile
            n = min(ranges[i - 1][1], b) - a
            w[:n] = torch.minimum(w[:n], (torch.arange(n) + 0.5) / n)
        if i + 1 < len(ranges) and ranges[i + 1][0] < b:        # and with the next one
            n = b - max(ranges[i + 1][0], a)
            w[b - a - n:] = torch.minimum(w[b - a - n:], ((torch.arange(n) + 0.5) / n).flip(0))
        ramps.append(w)
    return ramps


def resize_tile(tile, plan_x, plan_y, xr, yr, x0, y0):
    """The model output of one tile [n, C, h, w] taken to its target ranges xr, yr (model-output origin x0, y0)."""
    if plan_x is not None:
        index, w = plan_x
        index, w = index[xr[0]:xr[1]] - x0, w[xr[0]:xr[1]].to(tile.device, tile.dtype)
        index = index.clamp(0, tile.shape[-1] - 1).to(tile.device)
        tile = sum(tile.index_select(-1, index[:, t]) * w[:, t] for t in range(index.shape[1]))
    else:
        tile = tile[..., xr[0] - x0:xr[1] - x0]
    if plan_y is not None:
        index, w = plan_y
        index, w = index[yr[0]:yr[1]] - y0, w[yr[0]:yr[1]].to(tile.device, tile.dtype)
        index = index.clamp(0, tile.shape[-2] - 1).to(tile.device)
        tile = sum(tile.index_select(-2, index[:, t]) * w[:, t, None] for t in range(index.shape[1]))
    else:
        tile = tile[..., yr[0] - y0:yr[1] - y0, :]
    return tile


class UpscaleWithModelTiled:
    """Run an upscale model (Load Upscale Model) over an image batch tile by tile.

    tile / overlap are input px; tiles_per_batch tiles go through the model
    together. A quarter of the overlap at a tile's inner edges is left out
    (what the model makes there sees its padding), the rest is blended. precision fp16 / bf16 runs the model under autocast, and any tile
    that comes out NaN or infinite is run again in fp32. Tiles are blended with
    a linear ramp across their overlap and put together in float32 on
    output_device. downscale_to resizes every tile's result straight to the
    final size (Pillow's filters, the same weights as resizing the whole
    result), so the full model-size image is never held. Out of memory: the
    tile is halved and the frame run again.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "tile": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                 "tooltip": "Tile side in input px."}),
                "overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1,
                                    "tooltip": "Overlap between tiles in input px: a quarter of it at each inner "
                                               "edge is dropped (the model's edge effects), the rest blended with "
                                               "a linear ramp."}),
                "tiles_per_batch": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "precision": (PRECISIONS, {"default": "fp16",
                                           "tooltip": "Autocast precision; a NaN tile is run again in fp32."}),
                "output_device": (["cpu", "gpu"], {"default": "cpu",
                                                   "tooltip": "Where the result is put together."}),
                "downscale_to": (DOWNSCALE, {"default": "none",
                                             "tooltip": "Resize each tile's result straight to the final size: "
                                                        "width x height, or the input size x factor."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 1}),
                "factor": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 16.0, "step": 0.001,
                                     "tooltip": "downscale_to factor: the final size is the input size x this."}),
                "interpolation": (list(INTERPOLATIONS), {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "upscale"
    CATEGORY = f"{CATEGORY}/image"
    DESCRIPTION = "Upscale with a model, tile by tile in fp16 / bf16, optionally straight to a final size."

    def upscale(self, upscale_model, image, tile=1024, overlap=32, tiles_per_batch=4, precision="fp16",
                output_device="cpu", downscale_to="none", width=0, height=0, factor=2.0,
                interpolation="lanczos"):
        start = time.time()
        device = self._load(upscale_model, image, tile)
        scale = int(getattr(upscale_model, "scale", 1))
        batch, in_h, in_w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        out_w, out_h = in_w * scale, in_h * scale
        if downscale_to == "width_height":
            if width < 1 or height < 1:
                raise ValueError("downscale_to width_height needs width and height")
            target_w, target_h = int(width), int(height)
        elif downscale_to == "factor":
            target_w, target_h = max(1, round(in_w * factor)), max(1, round(in_h * factor))
        else:
            target_w, target_h = out_w, out_h
        kind = INTERPOLATIONS[interpolation] if (target_w, target_h) != (out_w, out_h) else None
        gather = torch.device("cpu") if output_device == "cpu" else device

        alpha = image[..., 3:4] if image.shape[-1] == 4 else None
        rgb = image[..., :3]
        size = int(tile)
        while True:
            try:
                result, stats = self._run(upscale_model, rgb, size, int(overlap), int(tiles_per_batch), precision,
                                          device, gather, scale, (target_w, target_h), kind)
                break
            except Exception as error:  # out of memory: smaller tiles
                if not self._is_oom(error) or size // 2 < 128:
                    raise
                size //= 2
                logging.warning("[Upscale With Model (tiled)] out of memory: tile %d px and again", size)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if alpha is not None:
            from .resample import resize as resample

            a = resample(alpha.float(), target_w, target_h, "bilinear")
            result = torch.cat([result, a.to(result.device, result.dtype).clamp(0, 1)], -1)

        seconds = (time.time() - start) / max(batch, 1)
        name = getattr(getattr(upscale_model, "architecture", None), "name", type(upscale_model).__name__)
        info = (f"{name} x{scale}, tile {size} overlap {overlap}, {stats['precision']}"
                + (f" ({stats['fp32_tiles']} tile(s) again in fp32: NaN)" if stats["fp32_tiles"] else "")
                + f", {stats['tiles']} tiles a frame, {seconds:.1f} s a frame, {target_w}x{target_h}"
                + ("" if kind is None else f" ({interpolation} from {out_w}x{out_h})"))
        logging.info("[Upscale With Model (tiled)] %s", info)
        return (result, info)

    @staticmethod
    def _is_oom(error):
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
        try:
            import comfy.model_management

            return isinstance(error, comfy.model_management.OOM_EXCEPTION)
        except Exception:
            return "out of memory" in str(error).lower()

    @staticmethod
    def _load(upscale_model, image, tile):
        """Put the model on its device the way ComfyUI does (kept there between runs) and return the device."""
        try:
            import comfy.model_management as mm

            patcher = getattr(upscale_model, "patcher", None)
            if patcher is not None:
                need = (tile * tile * 3) * 4 * max(getattr(upscale_model, "scale", 1), 1.0) * 384.0
                mm.load_models_gpu([patcher], memory_required=need, force_full_load=True)
                return patcher.load_device
            device = mm.get_torch_device()
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        upscale_model.to(device)
        return device

    @staticmethod
    def _run(model, rgb, tile, overlap, per_batch, precision, device, gather, scale, target, kind):
        batch, in_h, in_w, _ = rgb.shape
        out_w, out_h = in_w * scale, in_h * scale
        target_w, target_h = target
        tile_w, tile_h = min(tile, in_w), min(tile, in_h)
        overlap = min(overlap, max(tile_w, tile_h) - 1)
        xs, ys = tile_starts(in_w, tile_w, overlap), tile_starts(in_h, tile_h, overlap)
        plan_x = target_plan(out_w, target_w, kind)
        plan_y = target_plan(out_h, target_h, kind)
        margin = overlap // 4  # a tile's inner edges see the model's padding: dropped, not blended

        def inner(starts, size):
            last = len(starts) - 1
            return [((p + (margin if i > 0 else 0)) * scale, (p + size - (margin if i < last else 0)) * scale)
                    for i, p in enumerate(starts)]

        xr = [valid_range(plan_x, lo, hi, out_w) for lo, hi in inner(xs, tile_w)]
        yr = [valid_range(plan_y, lo, hi, out_h) for lo, hi in inner(ys, tile_h)]
        for ranges, starts, full in ((xr, xs, target_w), (yr, ys, target_h)):
            covered = torch.zeros(full, dtype=torch.bool)
            for a, b in ranges:
                covered[a:b] = True
            if not bool(covered.all()):
                raise ValueError("the overlap is too small for this resize: raise overlap")
        fx, fy = feather(xr), feather(yr)

        use_autocast = precision != "fp32" and device.type == "cuda"
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
        tiles = [(y, x, j, i) for j, y in enumerate(ys) for i, x in enumerate(xs)]
        try:
            import comfy.utils

            bar = comfy.utils.ProgressBar(batch * len(tiles))
        except Exception:
            bar = None
        fp32_tiles = 0
        frames = []
        for b in range(batch):
            acc = torch.zeros((3, target_h, target_w), dtype=torch.float32, device=gather)
            wsum = torch.zeros((target_h, target_w), dtype=torch.float32, device=gather)
            frame = rgb[b].movedim(-1, 0).to(device, torch.float32)
            for k in range(0, len(tiles), per_batch):
                group = tiles[k:k + per_batch]
                inp = torch.stack([frame[:, y:y + tile_h, x:x + tile_w] for y, x, _, _ in group])
                with torch.no_grad():
                    if use_autocast:
                        with torch.autocast(device.type, dtype=dtype):
                            out = model(inp)
                    else:
                        out = model(inp)
                    out = out.float()
                    bad = ~torch.isfinite(out).flatten(1).all(1)
                    if bool(bad.any()):                          # NaN / inf in half precision: fp32 for those
                        fp32_tiles += int(bad.sum())
                        out[bad] = model(inp[bad].float()).float()
                for (y, x, j, i), result in zip(group, out):
                    piece = resize_tile(result[None], plan_x, plan_y, xr[i], yr[j], x * scale, y * scale)[0]
                    w = fy[j][:, None] * fx[i][None, :]
                    a0, a1 = xr[i]
                    c0, c1 = yr[j]
                    acc[:, c0:c1, a0:a1] += (piece.clamp(0, 1) * w.to(piece.device)).to(gather)
                    wsum[c0:c1, a0:a1] += w.to(gather)
                if bar is not None:
                    bar.update(len(group))
            frames.append((acc / wsum.clamp(min=1e-8)).movedim(0, -1))
        stats = {"tiles": len(tiles), "fp32_tiles": fp32_tiles,
                 "precision": precision if use_autocast or precision == "fp32" else f"fp32 ({precision} on GPU only)"}
        return torch.stack(frames), stats


NODE_CLASS_MAPPINGS = {
    "ToolsUpscaleWithModelTiled": UpscaleWithModelTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsUpscaleWithModelTiled": "Upscale With Model (tiled, tools)",
}
