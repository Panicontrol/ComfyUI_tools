import pytest

from comfyui_tools.text_nodes import MAX_PARAGRAPHS, PromptParagraphs, unescape


def build(**kwargs):
    params = {
        "paragraphs": 4,
        "separator": "blank line",
        "custom_separator": "",
        "skip_empty": True,
        "strip": True,
        "comment_prefix": "//",
    }
    cells = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith("text_")}
    params.update(kwargs)
    return PromptParagraphs().build(**params, **cells)


def test_every_cell_is_declared_and_optional():
    types = PromptParagraphs.INPUT_TYPES()
    assert len(types["optional"]) == MAX_PARAGRAPHS
    assert types["optional"]["text_1"][1]["multiline"] is True
    assert types["required"]["paragraphs"][1]["max"] == MAX_PARAGRAPHS


def test_joins_the_cells_with_a_blank_line():
    text, used = build(text_1="first", text_2="second")
    assert text == "first\n\nsecond"
    assert used == 2


def test_only_the_active_cells_are_used():
    text, used = build(paragraphs=2, text_1="a", text_2="b", text_3="hidden")
    assert text == "a\n\nb"
    assert used == 2


def test_hidden_cells_come_back_when_the_count_grows():
    cells = {"text_1": "a", "text_2": "b", "text_3": "c"}
    assert build(paragraphs=1, **cells)[0] == "a"
    assert build(paragraphs=3, **cells)[0] == "a\n\nb\n\nc"


def test_separator_presets():
    cells = {"text_1": "a", "text_2": "b"}
    assert build(separator="new line", **cells)[0] == "a\nb"
    assert build(separator="space", **cells)[0] == "a b"
    assert build(separator="comma", **cells)[0] == "a, b"
    assert build(separator="none", **cells)[0] == "ab"


def test_custom_separator_unescapes():
    text, _ = build(separator="custom", custom_separator=" \\n-- ", text_1="a", text_2="b")
    assert text == "a \n-- b"


def test_unknown_separator_falls_back_to_a_blank_line():
    assert build(separator="???", text_1="a", text_2="b")[0] == "a\n\nb"


def test_empty_cells_are_skipped():
    text, used = build(text_1="a", text_2="   ", text_3="", text_4="b")
    assert text == "a\n\nb"
    assert used == 2


def test_empty_cells_can_be_kept():
    text, used = build(skip_empty=False, paragraphs=3, text_1="a", text_2="", text_3="b")
    assert text == "a\n\n\n\nb"
    assert used == 3


def test_a_cell_the_frontend_never_sent_is_skipped_either_way():
    # an unconnected optional input arrives as nothing at all, which is not
    # the same as an empty widget
    assert build(skip_empty=False, paragraphs=3, text_1="a", text_3="b") == ("a\n\nb", 2)


def test_strip_can_be_turned_off():
    assert build(strip=False, text_1="  a  ", text_2="b")[0] == "  a  \n\nb"


def test_comment_prefix_mutes_a_paragraph():
    text, used = build(text_1="a", text_2="// draft line", text_3="b")
    assert text == "a\n\nb"
    assert used == 2


def test_comment_prefix_can_be_changed_and_disabled():
    assert build(comment_prefix="#", text_1="# off", text_2="b")[0] == "b"
    assert build(comment_prefix="", text_1="// kept", text_2="b")[0] == "// kept\n\nb"
    assert build(comment_prefix="   ", text_1="// kept")[0] == "// kept"


def test_missing_cells_are_ignored():
    assert build(paragraphs=MAX_PARAGRAPHS, text_7="only me") == ("only me", 1)


def test_the_count_is_clamped_to_the_cells_that_exist():
    assert build(paragraphs=999, text_1="a")[0] == "a"
    assert build(paragraphs=0, text_1="a")[0] == "a"


def test_non_string_cell_values_are_accepted():
    assert build(text_1=42, text_2="b")[0] == "42\n\nb"


@pytest.mark.parametrize("raw,expected", [("a\\nb", "a\nb"), ("a\\tb", "a\tb"), ("ab", "ab")])
def test_unescape(raw, expected):
    assert unescape(raw) == expected
