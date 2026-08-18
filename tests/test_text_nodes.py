from comfyui_tools.text_nodes import TextCleanPrompt, TextConcat, TextTemplate


def test_concat_skips_empty_inputs():
    result, = TextConcat().concat(", ", True, "a portrait", "", None, "  8k  ")
    assert result == "a portrait, 8k"


def test_concat_can_keep_whitespace():
    result, = TextConcat().concat(" | ", False, " a ", "b")
    assert result == " a  | b"


def test_concat_unescapes_the_delimiter():
    result, = TextConcat().concat("\\n", True, "line one", "line two")
    assert result == "line one\nline two"


def test_template_fills_placeholders():
    result, = TextTemplate().render("{a} wearing {b}, {c}", "a knight", "red armor", "sunset")
    assert result == "a knight wearing red armor, sunset"


def test_template_treats_missing_values_as_empty():
    result, = TextTemplate().render("[{a}][{b}]", "x")
    assert result == "[x][]"


def test_clean_prompt_collapses_whitespace_and_duplicates():
    text, count = TextCleanPrompt().clean("a  cat ,, a cat,\n dog ", True, False)
    assert text == "a cat, dog"
    assert count == 2


def test_clean_prompt_can_keep_duplicates_and_lowercase():
    text, count = TextCleanPrompt().clean("Cat, cat", False, True)
    assert text == "cat, cat"
    assert count == 2
