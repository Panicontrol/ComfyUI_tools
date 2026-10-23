"""Registers H3 Masked Split Upscale and its split param nodes.

The implementation imports ComfyUI's sampling internals at module level, so it
lives in _h3_split_upscale and is only loaded inside ComfyUI; elsewhere (the
test suite), or on a ComfyUI too old for it, this pack simply has no such nodes
and every other node still loads.
"""

import logging

try:
    from . import _h3_split_upscale as _impl
except ImportError:  # pragma: no cover - outside ComfyUI
    _impl = None
except Exception as error:  # pragma: no cover - an incompatible ComfyUI
    logging.warning("[comfyui_tools] H3 Masked Split Upscale is unavailable: %s", error)
    _impl = None

NODE_CLASS_MAPPINGS = dict(_impl.NODE_CLASS_MAPPINGS) if _impl else {}
NODE_DISPLAY_NAME_MAPPINGS = dict(_impl.NODE_DISPLAY_NAME_MAPPINGS) if _impl else {}
