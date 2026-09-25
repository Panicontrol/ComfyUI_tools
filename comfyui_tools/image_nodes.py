"""Image utility nodes."""

import os
import re

import torch

from .utils import (
    CATEGORY,
    INTERPOLATION_MODES,
    hex_to_rgb,
    resize_image,
    resize_mask,
    round_to_multiple,
)

try:  # both ship with ComfyUI
    import numpy as np
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - exercised only without Pillow
    np = None
    Image = None
    ImageOps = None


def natural_key(name):
    """Sort key that orders frame_2 before frame_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def resolve_directory(directory):
    """Absolute path for a folder, allowing ~ and ComfyUI's input folder."""
    path = os.path.expanduser(str(directory).strip().strip('"'))
    if os.path.isabs(path):
        return path

    try:  # a relative path is most useful against ComfyUI's input folder
        import folder_paths

        candidate = os.path.join(folder_paths.get_input_directory(), path)
        if os.path.isdir(candidate):
            return candidate
    except Exception:
        pass
    return os.path.abspath(path)


def list_sequence(directory, pattern):
    """Files in ``directory`` matching any of the comma separated globs."""
    import fnmatch

    globs = [p.strip() for p in str(pattern).split(",") if p.strip()] or ["*"]
    names = []
    for name in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, name)):
            continue
        if any(fnmatch.fnmatch(name.lower(), g.lower()) for g in globs):
            names.append(name)
    return names


def load_frame(path):
    """One file as ``([1, H, W, 3], [1, H, W])`` image and mask tensors."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)

        if "A" in image.getbands():
            alpha = np.array(image.getchannel("A"), dtype=np.float32) / 255.0
            mask = torch.from_numpy(1.0 - alpha).unsqueeze(0)
        else:
            mask = None

        pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0

    frame = torch.from_numpy(pixels).unsqueeze(0)
    if mask is None:
        mask = torch.zeros((1, frame.shape[1], frame.shape[2]), dtype=torch.float32)
    return frame, mask


class ImageResize:
    """Resize an image by longest/shortest side, megapixels or explicit size."""

    MODES = ["longest_side", "shortest_side", "megapixels", "width_height"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "longest_side"}),
                "target_side": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 64.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "keep_aspect": ("BOOLEAN", {"default": True}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "interpolation": (INTERPOLATION_MODES, {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = f"{CATEGORY}/image"

    def target_size(self, src_w, src_h, mode, target_side, megapixels, width, height, keep_aspect):
        aspect = src_w / src_h
        if mode == "longest_side":
            if src_w >= src_h:
                return target_side, target_side / aspect
            return target_side * aspect, target_side
        if mode == "shortest_side":
            if src_w <= src_h:
                return target_side, target_side / aspect
            return target_side * aspect, target_side
        if mode == "megapixels":
            scale = ((megapixels * 1_000_000) / (src_w * src_h)) ** 0.5
            return src_w * scale, src_h * scale
        if not keep_aspect:
            return width, height
        scale = min(width / src_w, height / src_h)
        return src_w * scale, src_h * scale

    def resize(self, image, mode, target_side, megapixels, width, height,
               keep_aspect, multiple_of, interpolation):
        src_h, src_w = image.shape[1], image.shape[2]
        new_w, new_h = self.target_size(
            src_w, src_h, mode, target_side, megapixels, width, height, keep_aspect
        )
        new_w = round_to_multiple(new_w, multiple_of)
        new_h = round_to_multiple(new_h, multiple_of)
        return (resize_image(image, new_w, new_h, interpolation), new_w, new_h)


class ImagePadToRatio:
    """Pad an image to a target aspect ratio and return the padding as a mask."""

    POSITIONS = ["center", "top", "bottom", "left", "right"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ratio_width": ("FLOAT", {"default": 16.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "ratio_height": ("FLOAT", {"default": 9.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "position": (cls.POSITIONS, {"default": "center"}),
                "pad_color": ("STRING", {"default": "#000000"}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "pad_mask", "width", "height")
    FUNCTION = "pad"
    CATEGORY = f"{CATEGORY}/image"

    def offsets(self, position, extra_x, extra_y):
        left = extra_x // 2
        top = extra_y // 2
        if position == "left":
            left = 0
        elif position == "right":
            left = extra_x
        elif position == "top":
            top = 0
        elif position == "bottom":
            top = extra_y
        return left, top

    def pad(self, image, ratio_width, ratio_height, position, pad_color, multiple_of):
        batch, src_h, src_w, channels = image.shape
        ratio = ratio_width / ratio_height

        if src_w / src_h < ratio:
            new_w, new_h = src_h * ratio, float(src_h)
        else:
            new_w, new_h = float(src_w), src_w / ratio

        new_w = max(round_to_multiple(new_w, multiple_of), src_w)
        new_h = max(round_to_multiple(new_h, multiple_of), src_h)

        color = hex_to_rgb(pad_color)
        canvas = torch.empty(
            (batch, new_h, new_w, channels), dtype=image.dtype, device=image.device
        )
        for c in range(channels):
            canvas[..., c] = color[c] if c < len(color) else 1.0

        mask = torch.ones((batch, new_h, new_w), dtype=image.dtype, device=image.device)
        left, top = self.offsets(position, new_w - src_w, new_h - src_h)

        canvas[:, top:top + src_h, left:left + src_w, :] = image
        mask[:, top:top + src_h, left:left + src_w] = 0.0
        return (canvas, mask, new_w, new_h)


class ImageInfo:
    """Read the dimensions of an image batch as plain numbers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "batch_size", "aspect_ratio", "summary")
    FUNCTION = "info"
    CATEGORY = f"{CATEGORY}/image"

    def info(self, image):
        batch, height, width = image.shape[0], image.shape[1], image.shape[2]
        aspect = width / height
        summary = f"{width}x{height} | batch {batch} | aspect {aspect:.3f}"
        return (width, height, batch, aspect, summary)


class LoadImageSequence:
    """Load a folder of images -- a rendered PNG sequence -- as one IMAGE batch.

    The frames are ordered the way a sequence expects (``frame_2`` before
    ``frame_10``) and sliced with start/step/count, so a long render can be
    fed in pieces. A batch needs one size for every frame, so frames that do
    not match the first one are an error unless you ask for them to be resized
    or skipped.
    """

    SORT_MODES = ["name", "modified", "size"]
    MISMATCH = ["error", "resize", "skip"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "tooltip": "Folder holding the sequence. Absolute, ~ or relative to ComfyUI's input folder.",
                }),
                "pattern": ("STRING", {
                    "default": "*.png",
                    "tooltip": "Comma separated globs, e.g. '*.png' or 'frame_*.png, render_*.png'.",
                }),
                "sort_by": (cls.SORT_MODES, {
                    "default": "name",
                    "tooltip": "name sorts the way a sequence reads: frame_2 before frame_10.",
                }),
                "reverse": ("BOOLEAN", {"default": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "step": ("INT", {
                    "default": 1, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "Take every Nth frame.",
                }),
                "count": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "How many frames to load; 0 loads all of them.",
                }),
                "on_size_mismatch": (cls.MISMATCH, {
                    "default": "error",
                    "tooltip": "What to do with a frame that is not the size of the first one.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("images", "masks", "count", "filenames")
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/image"
    DESCRIPTION = "Load an image sequence from a folder as one batch, with the alpha channel as masks."

    @classmethod
    def sorted_names(cls, directory, pattern, sort_by, reverse):
        names = list_sequence(directory, pattern)
        if sort_by == "modified":
            names.sort(key=lambda name: os.path.getmtime(os.path.join(directory, name)))
        elif sort_by == "size":
            names.sort(key=lambda name: os.path.getsize(os.path.join(directory, name)))
        else:
            names.sort(key=natural_key)
        if reverse:
            names.reverse()
        return names

    @classmethod
    def IS_CHANGED(cls, directory, pattern, sort_by, reverse, start_index, step, count,
                   on_size_mismatch, **kwargs):
        # re-run when the folder's contents change, not just its path
        folder = resolve_directory(directory)
        if not os.path.isdir(folder):
            return f"missing:{folder}"
        state = [
            (name, os.path.getmtime(os.path.join(folder, name)), os.path.getsize(os.path.join(folder, name)))
            for name in sorted(list_sequence(folder, pattern))
        ]
        return str((folder, pattern, sort_by, reverse, start_index, step, count, on_size_mismatch, state))

    def load(self, directory, pattern, sort_by, reverse, start_index, step, count,
             on_size_mismatch):
        if Image is None:
            raise RuntimeError("this node needs Pillow and numpy (both ship with ComfyUI)")

        folder = resolve_directory(directory)
        if not directory or not str(directory).strip():
            raise ValueError("give the node a folder to load the sequence from")
        if not os.path.isdir(folder):
            raise ValueError(f"not a folder: {folder}")

        names = self.sorted_names(folder, pattern, sort_by, reverse)
        if not names:
            raise ValueError(f"no files matching {pattern!r} in {folder}")

        names = names[start_index::step]
        if count > 0:
            names = names[:count]
        if not names:
            raise ValueError(
                f"start_index {start_index} and step {step} skip past every file in {folder}"
            )

        frames = []
        masks = []
        loaded = []
        width = height = None
        for name in names:
            frame, mask = load_frame(os.path.join(folder, name))
            if width is None:
                height, width = frame.shape[1], frame.shape[2]
            elif frame.shape[2] != width or frame.shape[1] != height:
                if on_size_mismatch == "skip":
                    continue
                if on_size_mismatch == "error":
                    raise ValueError(
                        f"{name} is {frame.shape[2]}x{frame.shape[1]}, but the batch is "
                        f"{width}x{height}; set on_size_mismatch to resize or skip"
                    )
                frame = resize_image(frame, width, height, "lanczos")
                mask = resize_mask(mask, width, height)
            frames.append(frame)
            masks.append(mask)
            loaded.append(name)

        if not frames:
            raise ValueError(f"every frame after the first was skipped as the wrong size in {folder}")

        return (torch.cat(frames), torch.cat(masks), len(loaded), "\n".join(loaded))


NODE_CLASS_MAPPINGS = {
    "ToolsImageResize": ImageResize,
    "ToolsImagePadToRatio": ImagePadToRatio,
    "ToolsImageInfo": ImageInfo,
    "ToolsLoadImageSequence": LoadImageSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsImageResize": "Image Resize (tools)",
    "ToolsImagePadToRatio": "Image Pad To Ratio (tools)",
    "ToolsImageInfo": "Image Info (tools)",
    "ToolsLoadImageSequence": "Load Image Sequence (tools)",
}
