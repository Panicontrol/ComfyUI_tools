"""Text / prompt building nodes."""

from .utils import CATEGORY


class TextConcat:
    """Join up to four text inputs, skipping the empty ones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": ", "}),
                "strip_whitespace": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "text_a": ("STRING", {"forceInput": True}),
                "text_b": ("STRING", {"forceInput": True}),
                "text_c": ("STRING", {"forceInput": True}),
                "text_d": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concat"
    CATEGORY = f"{CATEGORY}/text"

    def concat(self, delimiter, strip_whitespace, text_a=None, text_b=None,
               text_c=None, text_d=None):
        delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
        parts = []
        for value in (text_a, text_b, text_c, text_d):
            if value is None:
                continue
            text = str(value)
            if strip_whitespace:
                text = text.strip()
            if text:
                parts.append(text)
        return (delimiter.join(parts),)


class TextTemplate:
    """Fill ``{a}``/``{b}``/``{c}`` placeholders in a template string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"default": "{a}, {b}, {c}", "multiline": True}),
            },
            "optional": {
                "a": ("STRING", {"default": "", "forceInput": True}),
                "b": ("STRING", {"default": "", "forceInput": True}),
                "c": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "render"
    CATEGORY = f"{CATEGORY}/text"

    def render(self, template, a=None, b=None, c=None):
        values = {"a": str(a or ""), "b": str(b or ""), "c": str(c or "")}
        text = template
        for key, value in values.items():
            text = text.replace("{" + key + "}", value)
        return (text,)


class TextCleanPrompt:
    """Normalise a prompt: collapse whitespace and drop duplicate/empty tags."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "remove_duplicates": ("BOOLEAN", {"default": True}),
                "lowercase": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "tag_count")
    FUNCTION = "clean"
    CATEGORY = f"{CATEGORY}/text"

    def clean(self, text, remove_duplicates, lowercase):
        raw = text.replace("\n", ",")
        tags = []
        seen = set()
        for chunk in raw.split(","):
            tag = " ".join(chunk.split())
            if not tag:
                continue
            if lowercase:
                tag = tag.lower()
            key = tag.lower()
            if remove_duplicates and key in seen:
                continue
            seen.add(key)
            tags.append(tag)
        return (", ".join(tags), len(tags))


MAX_PARAGRAPHS = 12

SEPARATOR_PRESETS = {
    "blank line": "\n\n",
    "new line": "\n",
    "space": " ",
    "comma": ", ",
    "none": "",
}


def unescape(text):
    """Turn the ``\\n`` / ``\\t`` a widget can carry into real characters."""
    return text.replace("\\n", "\n").replace("\\t", "\t")


class PromptParagraphs:
    """Write a prompt as separate paragraphs and join them into one string.

    Replaces a chain of text nodes feeding String Concatenate: every paragraph
    gets its own cell you can edit, reorder or comment out on its own. Only the
    first ``paragraphs`` cells are used, so raising and lowering that number
    shows and hides cells without losing what is written in them.
    """

    MAX = MAX_PARAGRAPHS
    SEPARATORS = list(SEPARATOR_PRESETS) + ["custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paragraphs": ("INT", {
                    "default": 4, "min": 1, "max": cls.MAX, "step": 1,
                    "tooltip": "How many cells are in use. The rest are ignored (and hidden in the UI).",
                }),
                "separator": (cls.SEPARATORS, {
                    "default": "blank line",
                    "tooltip": "What goes between the paragraphs.",
                }),
                "custom_separator": ("STRING", {
                    "default": "",
                    "tooltip": "Used when separator is 'custom'. \\n and \\t work here.",
                }),
                "skip_empty": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Leave empty cells out instead of joining blank paragraphs.",
                }),
                "strip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Trim whitespace around each paragraph.",
                }),
                "comment_prefix": ("STRING", {
                    "default": "//",
                    "tooltip": "A cell starting with this is left out -- a way to mute a paragraph. Empty disables it.",
                }),
            },
            # Widget values are restored by position in saved workflows, so a
            # new widget may only be appended after the cells, never inserted
            # among them -- see tests/test_prompt_paragraphs.py. Editor-only
            # settings belong in node properties instead (web/).
            "optional": {
                f"text_{i}": ("STRING", {"default": "", "multiline": True})
                for i in range(1, cls.MAX + 1)
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "used")
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/text"
    DESCRIPTION = ("Write a prompt as separate paragraphs (one cell each) and join them "
                   "into a single string.")

    def separator_text(self, separator, custom_separator):
        if separator == "custom":
            return unescape(custom_separator)
        return SEPARATOR_PRESETS.get(separator, "\n\n")

    def build(self, paragraphs, separator, custom_separator, skip_empty, strip,
              comment_prefix, **cells):
        count = max(1, min(int(paragraphs), self.MAX))
        prefix = comment_prefix.strip()

        parts = []
        for index in range(1, count + 1):
            value = cells.get(f"text_{index}")
            if value is None:
                continue
            text = str(value)
            if strip:
                text = text.strip()
            if prefix and text.lstrip().startswith(prefix):
                continue
            if skip_empty and not text.strip():
                continue
            parts.append(text)

        separator = self.separator_text(separator, custom_separator)
        return (separator.join(parts), len(parts))


NODE_CLASS_MAPPINGS = {
    "ToolsTextConcat": TextConcat,
    "ToolsTextTemplate": TextTemplate,
    "ToolsTextCleanPrompt": TextCleanPrompt,
    "ToolsPromptParagraphs": PromptParagraphs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsTextConcat": "Text Concat (tools)",
    "ToolsTextTemplate": "Text Template (tools)",
    "ToolsTextCleanPrompt": "Clean Prompt (tools)",
    "ToolsPromptParagraphs": "Prompt Paragraphs (tools)",
}
