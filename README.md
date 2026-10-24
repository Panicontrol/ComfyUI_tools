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
| **Save Image Sequence** | Save a render into the project folder `save_folder` as `<new_folder>_v004.mp4` (always, H.264, `frame_rate`), plus the PNG sequence in `<new_folder>_v004/<new_folder>_v004.01001.png` (`need_save`) and a zip of it next to the video, `<new_folder>_v004.zip` (`need_zip`). `number_version` steps up every run (*control after generate* → `increment`); frames start at `start_frame_index` (1001) with `padding_frame_name` digits (5); masks go into the PNG alpha; existing files are kept unless `overwrite` is on. The mp4 carries the workflow — drop it onto ComfyUI to load the graph back (`embed_workflow`); `save_workflow_json` also writes it as `<new_folder>_v004.json` next to the video. Connect a `latent` to save it next to the video too, as `<new_folder>_v004.latent` (MiniMax H3 video+audio included), for Load Latent From. Connect `notes` (e.g. the `info` of the graph's Shot Paths nodes joined by Text Concat with a `\n` delimiter) to record what the version is made of: the version name on the first line, then the notes, in the mp4's `notes` tag and in `<new_folder>_v004.notes.txt`; empty notes write neither. |

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

### tools/video

| Node | What it does |
| --- | --- |
| **Tools Video Reel: concat + stamp** | Concatenate mp4 clips into one reel at `output_path` (H.264, `crf`), with each clip's own text stamp in the top-left corner on a translucent box. `items` holds one block per clip, separated by a blank line: the clip's absolute path, then its stamp lines. Clips of another size are fitted into `width` × `height` on black; `frame_step` 2 with `frame_rate` 12 keeps every second frame. Frames are streamed, so long reels never sit in memory; an existing reel is kept unless `overwrite` is on. |

### tools/shot

Folders of a staged pipeline: `<root>/<shot>/_tmp/<stage>/<name>_vNNN.*` per stage, picked results in `<root>/<shot>/final/<shot>_final_vNNN/`. `base` is the shot's one versioned chain, `_tmp/base/<shot>_vNNN`: generation writes v001, and every detail pass (`02_track` → `03_detail` for one object, then the merge) reads the latest base and writes the next version; `05_upscale` and final take the latest. Per object: `02_track` and `03_detail` (`<shot>_<object>_vNNN`). The older `01_generate`, `04_merge` and `04_upscale` still work. A version's main file is its `.latent`; the mp4 is a preview that can always be decoded from it, so a version exists when either is there -- a base from elsewhere is just `_tmp/base/<shot>_v001.latent`, checked by H3 Latent Check in the graphs that take it.

| Node | What it does |
| --- | --- |
| **Shot Paths** | Every path one stage of one shot needs, set in one place: `save_folder` and `new_folder` (`<shot>` or `<shot>_<object>`) for Save Image Sequence, and the `source_video` / `source_latent` / `source_frames` of `source_stage` -- `source_version` 0 picks the latest one on disk (the node reruns when that folder changes) and returns the version it chose. Versions are found by their `.latent` and their `.mp4` alike; a missing source (neither file) gives an error listing the versions that exist, and the three paths come out as usual even when one of the files is missing. `stage` base with `source_stage` base is a detail pass: the source is the latest base, Save writes the next one in the same folder. base has no objects, so `object` / `source_object` are ignored there (and `info` says so). `info` is one line naming the file actually picked -- `base <- base shot_01_v002`, `03_detail <- 02_track shot_01_fly_v001`, `base <- none` -- with `(no mp4)` / `(no latent)` at the end when that file is missing, for Save Image Sequence's `notes`. Creates no folders. |
| **Shot Finalize** | Copies picked results (`items`, one per line: `base v003 -> final`, `base v001 -> generation`, `02_track v004 fly -> mask_fly`) into a new `<shot>_final_vNNN`, renamed: video, PNG folder (`copy_frames`), `.latent` (`copy_latents`), workflow `.json`, `.notes.txt`, plus a `manifest.md` with sizes, the models, seeds, steps, denoise and prompt texts read from each mp4's embedded prompt, and each result's notes (mp4 tag, else `.notes.txt`). A result with only its `.latent` brings that `.latent` (whatever `copy_latents` says) and the manifest notes `mp4 нет`. A `base vNNN` item also gets «История базы»: the notes of every base version from v001 up to it, in order (`нет файла` for a missing one). Everything is checked before anything is copied; existing final folders are never touched; `dry_run` (on by default) only shows the plan. `delete_tmp` removes `<shot>/_tmp` only after every copy worked and only when `confirm_delete` holds the shot name -- to the recycle bin when `send2trash` is installed. The report shows on the node. |

### tools/latent

| Node | What it does |
| --- | --- |
| **Save Latent To** | Save a `LATENT` to any folder (created if missing) as `<prefix>_00001.latent`, `_00002`, … Keeps everything in the latent: MiniMax H3 video+audio `NestedTensor`s, `noise_mask`, dtype. Also outputs the saved `path`. |
| **Load Latent From** | Load a `.latent` file from any path — absolute, or relative to ComfyUI's output or input folder — and rebuild the latent exactly. Also reads files saved by ComfyUI's own Save Latent. |
| **H3 Encode Still** | For trying latent img2img on MiniMax H3 stills: takes the still latent you would sample (e.g. Fizgig H3 Still Latent), encodes the image with the H3 video VAE at that latent's size and swaps it in as the video part, keeping the audio part H3 needs. Batch 1, as H3 requires; sample with denoise below 1. |
| **H3 Latent Mask Merge** | Merge two MiniMax H3 latents by a mask: black keeps `latent_a` (e.g. the plate encoded with the H3 VAE), white takes `latent_b` (the generation). The mask can be one frame or one per frame; it is area-downscaled to the latent grid and mapped onto H3's tokens (1, 4, 4, 4, 4 frames), `temporal` = `min` keeps `latent_a` for a whole token if any of its frames is black. Same video size required; audio from either latent. `set_noise_mask` attaches the mask so the next sampler redraws only the white area. `grow` (px, + widens the white area) and `feather` (px, Gaussian, 32 by default) shape the mask edge before it goes to the 16 px latent grid. |
| **H3 Latent Crop** | Crop a MiniMax H3 latent in space and time and keep it valid for H3: the box widens outward to the 32 px grid H3 patches on, `start_frame` snaps to a multiple of 17 and `length` to 17k + 5 frames, and the audio part is cut to the same frames. With a `mask` the box is the union of its white area over the cropped frames: `threshold`, `despeckle` (ignore stray specks), `padding` in pixels plus `padding_percent` of the object, then optionally grown to an `aspect_ratio` (as exact as the 32 px grid allows) and a `min_size` short side around the object's centre. `time_from_mask` takes the frames from the mask too -- only where it is white, plus `time_padding`. The box stays fixed for the whole piece: a latent is one rectangle. Outputs the piece, the `crop` for pasting back, the mask cut to the box, an `info` (crop and object size, how much of the crop the object fills, a warning below 512 px) and a `preview` with the object box (green) and crop box (yellow) drawn on a frame of `image`, or on the mask. |
| **H3 Latent Paste** | Put a piece from H3 Latent Crop back into the full latent at the same size. Pastes the whole box, or only a mask's white area (full-frame or crop-sized); `feather` fades the edge inside the pasted area (capped at a quarter of the box), so nothing outside the box changes. Audio from the full latent or the piece. |
| **H3 Latent Diff Mask** | Find what a prompt edit changed: compare the source and edited MiniMax H3 latents (same size and frame count) per 16 px cell, after removing the overall colour drift regeneration adds and scaling each channel by the source's spread. Threshold automatic (Otsu) or manual, specks and holes cleaned, smoothed over time, then `grow` / `feather`. Outputs a per-frame `mask` (white = changed) for H3 Latent Mask Merge, a `heatmap` of the change score at 1/4 size, and an `info` line with the changed share. |
| **H3 Decode Frames** | Decode only the frames you want to look at from a MiniMax H3 latent (`0, 24, 48-60`, `0-123:24` every 24th, `-1` the last). Only the 17-frame groups holding them are decoded; with `exact` the group before is decoded too as context, so the frames match a full VAE Decode exactly. |
| **Latent Info** | Everything that can be read from a latent, shown on the node after a run: type (MiniMax H3 video+audio, video, image), size in pixels and latent cells, frames, tokens and seconds, whether the length is on H3's 17k+5 grid and the sides on its 32 px grid, the audio part's length, dtype and device, value statistics with NaN/Inf and all-zero latents flagged, `noise_mask` coverage and any other keys. The latent passes through; `info`, `frames`, `width`, `height` are outputs too. |
| **H3 Latent Check** | Stops the graph before decoding or sampling when a MiniMax H3 latent is not what it expects: size in px (`width` / `height`, default 1344x768, 0 = any), video `frames` (0 = any), frames on the 17k+5 grid (`frame_grid`) and an audio part (`require_audio`), read the way Latent Info reads them. Any mismatch is an error listing all of them, e.g. `latent check: 1280x720 (expected 1344x768); 120 frames not on 17k+5 grid; no audio part`. The latent passes through unchanged, with a one-line `info` (`1344x768, 124 frames (5.17 s @24), audio yes`), shown on the node too. |
| **H3 Latent Resize** | Resize a MiniMax H3 latent in space, down as well as up (the H3 Latent Upscaler models only go up). Only the video part is resized, frame by frame; audio and frame count stay, a `noise_mask` follows the video, and the size snaps to the 32 px grid. By `scale`, or `width` / `height` in px (one of them keeps the aspect). `area` averages cells and suits downscaling. Resizing latents is an approximation: decode / resize / encode, or a low-denoise pass, gives the cleanest result. |
| **H3 Masked Split Upscale** | Tiled MiniMax H3 latent refine (split in time and space, anchors, colour match, seam polish) that redraws only a `mask`: white = redraw, black = keep the input latent, re-injected at every sampling step and restored exactly at the end. `mask_grow` (32 px) and `mask_feather` (32 px Gaussian radius, a ~64 px hand-over), in pixels of the latent's frame whatever the mask's size, widen and soften the mask's edge first; with grow >= feather the object's own edge stays fully redrawn and the hand-over to the plate falls on the background around it. `skip_frozen_tiles` does not sample tiles and seams the mask leaves all black -- same result, faster when the masked object is small. `tiles` = `around mask` drops the frame grid: per temporal chunk every white blob of the mask (plus `context` px) gets one `tile_size` square centred on it, anywhere on the 32 px grid; blobs whose squares would overlap are merged, and a blob too big for one square gets its own small grid with the spatial params' overlap and fade. Use with **H3 Temporal / Spatial Split Params (tools)**; the split param nodes of the original pack plug in too. Based on MMH3 Split Upscale from [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) by LBH-123-AI (MIT), kept here under its own node ids so it does not depend on that pack's updates. Loads only inside ComfyUI. |

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
