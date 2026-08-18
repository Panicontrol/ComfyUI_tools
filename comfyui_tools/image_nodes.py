"""Image utility nodes."""

import torch

from .utils import (
    CATEGORY,
    INTERPOLATION_MODES,
    hex_to_rgb,
    resize_image,
    round_to_multiple,
)


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


NODE_CLASS_MAPPINGS = {
    "ToolsImageResize": ImageResize,
    "ToolsImagePadToRatio": ImagePadToRatio,
    "ToolsImageInfo": ImageInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsImageResize": "Image Resize (tools)",
    "ToolsImagePadToRatio": "Image Pad To Ratio (tools)",
    "ToolsImageInfo": "Image Info (tools)",
}
