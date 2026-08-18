"""Logic and workflow helper nodes."""

from .utils import ANY, CATEGORY, round_to_multiple


class SwitchAny:
    """Pass through one of two inputs of any type."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_first": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "on_true": (ANY, {"lazy": True}),
                "on_false": (ANY, {"lazy": True}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = f"{CATEGORY}/logic"

    def check_lazy_status(self, use_first, on_true=None, on_false=None):
        needed = "on_true" if use_first else "on_false"
        return [needed]

    def switch(self, use_first, on_true=None, on_false=None):
        return (on_true if use_first else on_false,)


class ResolutionPreset:
    """Common generation resolutions with orientation and scaling."""

    PRESETS = {
        "SD1.5 512x512": (512, 512),
        "SD1.5 768x512": (768, 512),
        "SDXL 1024x1024": (1024, 1024),
        "SDXL 1152x896": (1152, 896),
        "SDXL 1216x832": (1216, 832),
        "SDXL 1344x768": (1344, 768),
        "SDXL 1536x640": (1536, 640),
        "HD 1280x720": (1280, 720),
        "FullHD 1920x1080": (1920, 1080),
        "4K 3840x2160": (3840, 2160),
    }

    ORIENTATIONS = ["landscape", "portrait"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {"default": "SDXL 1024x1024"}),
                "orientation": (cls.ORIENTATIONS, {"default": "landscape"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "label")
    FUNCTION = "resolve"
    CATEGORY = f"{CATEGORY}/logic"

    def resolve(self, preset, orientation, scale, multiple_of):
        width, height = self.PRESETS[preset]
        if orientation == "portrait":
            width, height = height, width
        width = round_to_multiple(width * scale, multiple_of)
        height = round_to_multiple(height * scale, multiple_of)
        return (width, height, f"{width}x{height}")


class SeedRange:
    """Turn one seed into a small set of derived, deterministic seeds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "offset": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("seed", "seed_1", "seed_2", "seed_3")
    FUNCTION = "derive"
    CATEGORY = f"{CATEGORY}/logic"

    def derive(self, seed, offset):
        limit = 0xFFFFFFFFFFFFFFFF
        return tuple((seed + offset * i) % limit for i in range(4))


NODE_CLASS_MAPPINGS = {
    "ToolsSwitchAny": SwitchAny,
    "ToolsResolutionPreset": ResolutionPreset,
    "ToolsSeedRange": SeedRange,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsSwitchAny": "Switch Any (tools)",
    "ToolsResolutionPreset": "Resolution Preset (tools)",
    "ToolsSeedRange": "Seed Range (tools)",
}
