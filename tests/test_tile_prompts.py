import pytest
import torch

from comfyui_tools.tile_prompts import cell_at, draw_tile_map, fill_template, parse_tile_prompts, with_tile_text


def test_parse_tile_prompts_keys_priority_and_continuation():
    text = """# a 2 x 3 grid
*: plaster wall, soft daylight
r0c*: sky over the roofs
r0c1: a fly on the window glass,
  compound eyes, fine hairs
r1c0, r1c2: wooden table
R1C1:
"""
    got = parse_tile_prompts(text, 2, 3)
    assert got == {
        (0, 0): "sky over the roofs",
        (0, 1): "a fly on the window glass,\ncompound eyes, fine hairs",
        (0, 2): "sky over the roofs",
        (1, 0): "wooden table",
        (1, 2): "wooden table",
    }   # r1c1 is empty: it goes back to the upscale's conditioning
    # the more specific key wins whatever the order; for the same key the later line
    assert parse_tile_prompts("r0c0: a\n*: b\nr0c0: c", 1, 2) == {(0, 0): "c", (0, 1): "b"}
    assert parse_tile_prompts("# only comments\n\n", 2, 2) == {}


def test_parse_tile_prompts_errors():
    with pytest.raises(ValueError, match=r"r2c0 is outside the grid, which has 2 rows x 3 columns \(r0c0 ... r1c2\)"):
        parse_tile_prompts("r2c0: floor", 2, 3)
    with pytest.raises(ValueError, match="line 1: start with a tile"):
        parse_tile_prompts("a fly on the glass", 2, 3)


def test_fill_template():
    assert fill_template("macro lens, {tile}, film grain", "a fly") == "macro lens, a fly, film grain"
    assert fill_template("macro lens.", "a fly") == "macro lens.\na fly"
    assert fill_template("{tile}", "a fly") == "a fly"


def test_with_tile_text_keeps_everything_but_the_text():
    main = [[torch.zeros(1, 5, 4), {"pooled_output": "main pooled", "minimax_token_tags": "main tags",
                                    "minimax_keyframes": ["kf"], "minimax_visual_cond_noise_aug": 0.999}]]
    tile = [[torch.ones(1, 7, 4), {"pooled_output": None, "minimax_token_tags": "tile tags"}]]
    (tensor, d), = with_tile_text(main, tile)
    assert tensor.shape == (1, 7, 4) and d == {"pooled_output": None, "minimax_token_tags": "tile tags",
                                               "minimax_keyframes": ["kf"], "minimax_visual_cond_noise_aug": 0.999}
    (_, d), = with_tile_text(main, [[torch.ones(1, 7, 4), {}]])
    assert "minimax_token_tags" not in d and "pooled_output" not in d   # the main text's keys never leak
    assert main[0][1]["minimax_token_tags"] == "main tags"


def test_cell_at_and_tile_map():
    assert [cell_at([0, 40, 80], [48, 48, 40], p) for p in (0, 30, 64, 70, 90, 119)] == [0, 0, 1, 1, 2, 2]   # centres 24, 64, 100
    rows, cols, trows, tcols = [0, 40], [0, 40, 80], [48, 56], [48, 48, 40]   # latent cells of a 1920x1536 frame
    grey = draw_tile_map(None, 96, 120, rows, cols, trows, tcols, {(0, 1): 1})
    assert grey.shape == (1, 1229, 1536, 3) and float(grey.max()) <= 1.0
    frame = torch.full((3, 270, 480, 3), 0.5)
    drawn = draw_tile_map(frame, 96, 120, rows, cols, trows, tcols, {(0, 1): 1})
    assert drawn.shape == (1, 270, 480, 3)
    green = (drawn[0, :, :, 1] > 0.8) & (drawn[0, :, :, 0] < 0.4)
    ys, xs = green.nonzero(as_tuple=True)
    assert 160 - 4 <= int(xs.min()) and int(xs.max()) <= 352 + 4   # only r0c1 (cells 40-88 of 120) is green
