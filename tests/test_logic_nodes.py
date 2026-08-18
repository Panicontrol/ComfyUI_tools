from comfyui_tools.logic_nodes import ResolutionPreset, SeedRange, SwitchAny


def test_switch_picks_the_requested_branch():
    assert SwitchAny().switch(True, "yes", "no") == ("yes",)
    assert SwitchAny().switch(False, "yes", "no") == ("no",)


def test_switch_only_requests_the_branch_it_needs():
    assert SwitchAny().check_lazy_status(True) == ["on_true"]
    assert SwitchAny().check_lazy_status(False) == ["on_false"]


def test_resolution_preset_landscape():
    width, height, label = ResolutionPreset().resolve("SDXL 1216x832", "landscape", 1.0, 8)
    assert (width, height) == (1216, 832)
    assert label == "1216x832"


def test_resolution_preset_portrait_swaps_the_sides():
    width, height, _ = ResolutionPreset().resolve("SDXL 1216x832", "portrait", 1.0, 8)
    assert (width, height) == (832, 1216)


def test_resolution_preset_scale_is_rounded_to_multiple():
    width, height, _ = ResolutionPreset().resolve("SD1.5 512x512", "landscape", 1.5, 64)
    assert (width, height) == (768, 768)
    width, height, _ = ResolutionPreset().resolve("HD 1280x720", "landscape", 0.33, 8)
    assert width % 8 == 0 and height % 8 == 0


def test_seed_range_is_deterministic():
    assert SeedRange().derive(100, 5) == (100, 105, 110, 115)


def test_seed_range_with_zero_offset_repeats_the_seed():
    assert SeedRange().derive(7, 0) == (7, 7, 7, 7)
