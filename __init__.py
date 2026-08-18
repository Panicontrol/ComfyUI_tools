"""ComfyUI_tools -- a small collection of custom nodes for ComfyUI.

ComfyUI imports this package from ``custom_nodes/`` and reads the two mapping
dictionaries below to register every node in the pack.
"""

from .comfyui_tools import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
