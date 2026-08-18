"""Node registry for ComfyUI_tools.

Every module in this package whose name ends with ``_nodes`` is imported
automatically and its ``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS``
are merged here, so adding a node pack means dropping in a new file.
"""

import importlib
import pkgutil

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load_node_modules():
    for module_info in pkgutil.iter_modules(__path__):
        if not module_info.name.endswith("_nodes"):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        classes = getattr(module, "NODE_CLASS_MAPPINGS", {})
        names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})

        duplicates = set(classes) & set(NODE_CLASS_MAPPINGS)
        if duplicates:
            raise RuntimeError(
                f"duplicate node ids in {module_info.name}: {sorted(duplicates)}"
            )

        NODE_CLASS_MAPPINGS.update(classes)
        NODE_DISPLAY_NAME_MAPPINGS.update(names)


_load_node_modules()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
