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


NODE_CLASS_MAPPINGS = {
    "ToolsTextConcat": TextConcat,
    "ToolsTextTemplate": TextTemplate,
    "ToolsTextCleanPrompt": TextCleanPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsTextConcat": "Text Concat (tools)",
    "ToolsTextTemplate": "Text Template (tools)",
    "ToolsTextCleanPrompt": "Clean Prompt (tools)",
}
