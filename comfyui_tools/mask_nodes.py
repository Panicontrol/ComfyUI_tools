"""Mask utility nodes."""

import torch
import torch.nn.functional as F

from .utils import CATEGORY, gaussian_blur


class MaskGrowFeather:
    """Grow or shrink a mask, then soften its edge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = f"{CATEGORY}/mask"

    def process(self, mask, expand, feather, invert):
        samples = mask.unsqueeze(1).float()

        if expand != 0:
            size = abs(int(expand)) * 2 + 1
            if expand > 0:
                samples = F.max_pool2d(samples, kernel_size=size, stride=1, padding=size // 2)
            else:
                samples = -F.max_pool2d(-samples, kernel_size=size, stride=1, padding=size // 2)

        if feather > 0:
            samples = gaussian_blur(samples, int(feather))

        samples = samples.squeeze(1).clamp(0.0, 1.0)
        if invert:
            samples = 1.0 - samples
        return (samples,)


class MaskCombine:
    """Combine two masks with a boolean-style operation."""

    OPERATIONS = ["union", "intersection", "difference", "add", "multiply", "xor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "operation": (cls.OPERATIONS, {"default": "union"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine"
    CATEGORY = f"{CATEGORY}/mask"

    def combine(self, mask_a, mask_b, operation, strength):
        a = mask_a.float()
        b = mask_b.float() * strength

        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(
                b.unsqueeze(1), size=a.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)

        if operation == "union":
            result = torch.maximum(a, b)
        elif operation == "intersection":
            result = torch.minimum(a, b)
        elif operation == "difference":
            result = a - b
        elif operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        else:  # xor
            result = torch.abs(a - b)
        return (result.clamp(0.0, 1.0),)


class MaskBoundingBox:
    """Report the bounding box of the non-empty area of a mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("x", "y", "width", "height", "coverage")
    FUNCTION = "bounds"
    CATEGORY = f"{CATEGORY}/mask"

    def bounds(self, mask, threshold, padding):
        merged = mask.float().amax(dim=0)
        height, width = merged.shape
        hits = merged > threshold
        coverage = float(hits.float().mean().item())

        if not bool(hits.any()):
            return (0, 0, 0, 0, coverage)

        rows = torch.nonzero(hits.any(dim=1)).flatten()
        cols = torch.nonzero(hits.any(dim=0)).flatten()
        y0 = max(int(rows[0].item()) - padding, 0)
        y1 = min(int(rows[-1].item()) + 1 + padding, height)
        x0 = max(int(cols[0].item()) - padding, 0)
        x1 = min(int(cols[-1].item()) + 1 + padding, width)
        return (x0, y0, x1 - x0, y1 - y0, coverage)


NODE_CLASS_MAPPINGS = {
    "ToolsMaskGrowFeather": MaskGrowFeather,
    "ToolsMaskCombine": MaskCombine,
    "ToolsMaskBoundingBox": MaskBoundingBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsMaskGrowFeather": "Mask Grow / Feather (tools)",
    "ToolsMaskCombine": "Mask Combine (tools)",
    "ToolsMaskBoundingBox": "Mask Bounding Box (tools)",
}
