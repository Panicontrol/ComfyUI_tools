# ComfyUI_tools

A small pack of quality-of-life custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI):
image sizing, mask editing, prompt building, AV inpainting helpers and workflow logic — the glue nodes you end up
rebuilding in every workflow.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Panicontrol/ComfyUI_tools.git
```

Restart ComfyUI. The nodes appear under the **tools** category in the node menu.
There are no extra dependencies — the pack only uses `torch`, which ComfyUI already ships.

## Nodes

### tools/image

| Node | What it does |
| --- | --- |
| **Image Resize** | Resize by longest side, shortest side, megapixels or explicit width/height. Keeps aspect ratio, snaps to a multiple of N (8 by default) and also outputs the resulting width/height as `INT`. |
| **Image Pad To Ratio** | Pad an image out to a target aspect ratio with a chosen color and position. Returns the padded image plus a mask of the added area — ready to feed an outpainting pass. |
| **Image Info** | Width, height, batch size, aspect ratio and a printable summary of an image batch. |

### tools/mask

| Node | What it does |
| --- | --- |
| **Mask Grow / Feather** | Grow or shrink a mask by N pixels, soften the edge with a Gaussian feather, optionally invert. |
| **Mask Combine** | Union, intersection, difference, add, multiply or xor of two masks, with a strength factor for the second one. Mismatched sizes are resampled automatically. |
| **Mask Bounding Box** | Bounding box (x, y, width, height) of the non-empty area of a mask, with optional padding, plus the mask's coverage ratio. |

### tools/av

| Node | What it does |
| --- | --- |
| **Audio Mask** | Builds the `audio_mask` that [LanPaint](https://github.com/scraed/LanPaint) **AV Encode** requires: a `MASK` of shape `[F]` at the video frame rate, `1` = regenerate the audio at that moment, `0` = keep it. Defaults to an all-zero stub for video-only inpainting; can also regenerate everything or only the time ranges you list. |
| **Video Add Silent Audio** | Gives a soundless video a silent audio track exactly as long as the picture, so AV Encode stops raising *"the video has no audio track to encode"*. A video that already has audio is passed through untouched unless you ask for a replacement. |
| **Silent Audio** | A standalone silent `AUDIO` track, sized from a video, a frame count or plain seconds. |

Frame count and fps for the mask are taken from the connected `video`, or from
a per-frame `video_mask` (whose frame count wins), or from the widgets when
nothing is connected — so the mask always matches the clip you feed AV Encode.

Video-only inpainting on a clip with no sound at all:

```
Video -> Video Add Silent Audio -> LanPaint AV Encode (video)
mask -> LanPaint AV Encode (mask)
Audio Mask, mode keep_all -> LanPaint AV Encode (audio_mask)
```

`Audio Mask` modes: `keep_all` (zeros — the stub), `regenerate_all` (ones),
`intervals` (seconds, one range per line: `0.5-2.0`; the editor's
`[{"start": 0.5, "end": 2.0}]` JSON is accepted too, with the same frame
rounding LanPaint uses).

Both silence nodes default to a 32 kHz sample rate, which is what the MiniMax
H3 audio VAE runs at, so LanPaint never has to resample (that path needs
`torchaudio`).

### tools/text

| Node | What it does |
| --- | --- |
| **Text Concat** | Join up to four text inputs with a delimiter, skipping empty ones (`\n` and `\t` work in the delimiter). |
| **Text Template** | Fill `{a}`, `{b}`, `{c}` placeholders in a multiline template. |
| **Clean Prompt** | Collapse whitespace, drop empty and duplicate tags, optionally lowercase; also returns the tag count. |

### tools/logic

| Node | What it does |
| --- | --- |
| **Switch Any** | Route one of two inputs of any type through a boolean. Lazy — only the selected branch is evaluated. |
| **Resolution Preset** | Common SD1.5 / SDXL / HD resolutions with orientation, scaling and multiple-of rounding. |
| **Seed Range** | Derive three extra deterministic seeds from one seed and an offset. |

## Adding a node

1. Create `comfyui_tools/<something>_nodes.py`.
2. Write a class with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION` and `CATEGORY`.
3. Export `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` from the module.

Any module whose name ends with `_nodes` is discovered and merged automatically by
`comfyui_tools/__init__.py`; duplicate node ids raise on load.

## Tests

```bash
pip install pytest
python -m pytest
```

The suite runs standalone — it does not need a ComfyUI checkout, only `torch`.

## License

MIT — see [LICENSE](LICENSE).
