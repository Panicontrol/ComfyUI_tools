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
| **Load Image Sequence** | Load a folder of images — a rendered PNG sequence — as one `IMAGE` batch, with the alpha channel as `MASK`s. |
| **Extend Sequence** | Stretch a batch to a minimum length by mirroring, looping or holding the last frame — e.g. a 98-frame render to the 124 frames a model requires. |
| **Restore Sequence Length** | The reverse of Extend Sequence: cut the generated batch back to the original frame count before saving. Connect the pre-extension batch as `original`, or set `length`. |
| **Save Image Sequence** | Save a render into the project folder `save_folder` as `<new_folder>_v004.mp4` (always, H.264, `frame_rate`), plus the PNG sequence in `<new_folder>_v004/<new_folder>_v004.01001.png` (`need_save`) and a zip of it next to the video, `<new_folder>_v004.zip` (`need_zip`). `number_version` steps up every run (*control after generate* → `increment`); frames start at `start_frame_index` (1001) with `padding_frame_name` digits (5); masks go into the PNG alpha; existing files are kept unless `overwrite` is on. |

**Load Image Sequence** takes a folder path (absolute, `~/…`, or relative to
ComfyUI's input folder) and a `pattern` of comma separated globs (`*.png` by
default, `frame_*.png, render_*.png` also works). Frames are ordered the way a
sequence reads — `frame_2` before `frame_10` — or by modification time or size,
optionally reversed. `start_index`, `step` and `count` slice a long render into
pieces (every 2nd frame, the next 16, …). Since a batch needs one size, a frame
that does not match the first one is an error unless `on_size_mismatch` is set
to `resize` or `skip`. Outputs the batch, the masks, the frame `count` and the
`filenames` that went in, and it re-runs when the folder's contents change, not
just when its path does.

**Extend Sequence** takes any `IMAGE` batch (and optionally its `MASK`s) and a
target `length`. `mirror` plays the frames forward and then back without
repeating the frame it turns on (`0 1 2 3 2 1 0 1 …`), so 98 frames extended to
124 are frames 0–97 followed by 96 down to 71. `loop` starts over from frame 0
and `hold_last` freezes on the final frame. A batch that is already long enough
passes through whole unless `trim_longer` cuts it to exactly `length`. Besides
the batch and masks it outputs the `count` and `frame_order`, the source index
of every output frame.

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

### tools/latent

| Node | What it does |
| --- | --- |
| **Save Latent To** | Save a `LATENT` to any folder (created if missing) as `<prefix>_00001.latent`, `_00002`, … Keeps everything in the latent: MiniMax H3 video+audio `NestedTensor`s, `noise_mask`, dtype. Also outputs the saved `path`. |
| **Load Latent From** | Load a `.latent` file from any path — absolute, or relative to ComfyUI's output or input folder — and rebuild the latent exactly. Also reads files saved by ComfyUI's own Save Latent. |

### tools/text

| Node | What it does |
| --- | --- |
| **Text Concat** | Join up to four text inputs with a delimiter, skipping empty ones (`\n` and `\t` work in the delimiter). |
| **Text Template** | Fill `{a}`, `{b}`, `{c}` placeholders in a multiline template. |
| **Clean Prompt** | Collapse whitespace, drop empty and duplicate tags, optionally lowercase; also returns the tag count. |
| **Prompt Paragraphs** | Write a prompt as separate paragraphs, one editable cell each, and get a single `STRING` out — one node in place of a chain of text nodes feeding String Concatenate. |

**Prompt Paragraphs** has 12 cells; the `paragraphs` widget sets how many are in
use, and a small frontend script hides the rest. Hiding is cosmetic — the text
stays in the widget, so lowering the count and raising it again brings the
paragraph back. Cells also accept `STRING` links, so a shared paragraph can come
from another node.

Cells size themselves to their own text: a one-line paragraph gets a one-line
cell, a long one grows up to 20 lines and then scrolls. To make every cell the
same height instead, right click the node → **Cell height** and give it a line
count (0 goes back to fitting the text). That setting is stored as a node
property, not a widget, so it travels with the workflow without disturbing the
cells.

Other widgets: `separator` (blank line, new line, space, comma, none, custom —
`\n` and `\t` work in `custom_separator`), `skip_empty`, `strip`, and
`comment_prefix` (default `//`) which mutes any cell that starts with it — handy
for parking a paragraph while you iterate. Outputs the joined `text` and `used`,
the number of paragraphs that actually made it in.

### tools/logic

| Node | What it does |
| --- | --- |
| **Switch Any** | Route one of two inputs of any type through a boolean. Lazy — only the selected branch is evaluated. |
| **Resolution Preset** | Common SD1.5 / SDXL / HD resolutions with orientation, scaling and multiple-of rounding. |
| **Seed Range** | Derive three extra deterministic seeds from one seed and an offset. |
| **Version Counter** | A version number that goes up by one on every run — ComfyUI's own *control after generate* switch, set to `increment` by default. Outputs the `INT` and a `label` such as `v004` (`prefix` + zero `padding`). |

## Adding a node

1. Create `comfyui_tools/<something>_nodes.py`.
2. Write a class with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION` and `CATEGORY`.
3. Export `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` from the module.

Any module whose name ends with `_nodes` is discovered and merged automatically by
`comfyui_tools/__init__.py`; duplicate node ids raise on load.

Browser-side scripts live in `web/`, which `WEB_DIRECTORY` serves at
`/extensions/ComfyUI_tools/`. Keep them in that folder's root — a script one
level deeper resolves `../../scripts/app.js` to `/extensions/scripts/app.js`
and never loads — and keep them cosmetic: every node must still work with the
script missing.

Widget order is part of a node's contract: ComfyUI restores widget values by
position, so a widget added among the existing ones shifts every value after it
in workflows people have already saved. Append new widgets after the old ones,
and keep editor-only settings in node properties instead.

## Tests

```bash
pip install pytest
python -m pytest
```

The suite runs standalone — it does not need a ComfyUI checkout, only `torch`.

## License

MIT — see [LICENSE](LICENSE).
