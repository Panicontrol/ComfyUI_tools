import comfyui_tools


def test_registry_is_not_empty():
    assert comfyui_tools.NODE_CLASS_MAPPINGS


def test_every_node_has_a_display_name():
    missing = set(comfyui_tools.NODE_CLASS_MAPPINGS) - set(
        comfyui_tools.NODE_DISPLAY_NAME_MAPPINGS
    )
    assert not missing


def test_node_contract():
    for node_id, node_class in comfyui_tools.NODE_CLASS_MAPPINGS.items():
        assert node_id.startswith("Tools"), node_id
        assert isinstance(node_class.INPUT_TYPES(), dict), node_id
        assert node_class.RETURN_TYPES, node_id
        assert callable(getattr(node_class, node_class.FUNCTION)), node_id
        assert node_class.CATEGORY.startswith("tools/"), node_id
        if hasattr(node_class, "RETURN_NAMES"):
            assert len(node_class.RETURN_NAMES) == len(node_class.RETURN_TYPES), node_id


def test_all_node_packs_are_discovered():
    categories = {
        node_class.CATEGORY for node_class in comfyui_tools.NODE_CLASS_MAPPINGS.values()
    }
    assert categories == {"tools/image", "tools/mask", "tools/text", "tools/logic"}
