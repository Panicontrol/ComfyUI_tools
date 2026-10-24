"""Per-tile prompts for H3 Masked Split Upscale: which text each grid tile gets.

Tiles are named r<row>c<col> from the top left (r0c0, r0c1, ... r1c0, ...),
in the grid the upscale builds from the latent and its spatial_split_param.
Kept free of ComfyUI imports so it can be tested; the nodes live in
_h3_split_upscale.
"""

import re

import torch

KEY = r"(?:\*|r(?:\d+|\*)c(?:\d+|\*))"
LINE_RE = re.compile(rf"^\s*({KEY}(?:\s*,\s*{KEY})*)\s*:(.*)$", re.IGNORECASE)
# keys a text encoder fills in; everything else in a conditioning (keyframes, refs, anchors) is kept
TEXT_KEYS = ("pooled_output", "minimax_token_tags", "attention_mask")


def tile_name(ri, cj):
    return f"r{ri}c{cj}"


def _key_tiles(key, nrows, ncols):
    """The tiles one key names, and its priority (fixed coordinates: r0c1 2, r0c* 1, * 0)."""
    key = key.lower()
    if key == "*":
        return [(r, c) for r in range(nrows) for c in range(ncols)], 0
    r, c = re.fullmatch(r"r(\d+|\*)c(\d+|\*)", key).groups()
    for value, count, what in ((r, nrows, "rows"), (c, ncols, "columns")):
        if value != "*" and int(value) >= count:
            raise ValueError(f"tile prompts: {key} is outside the grid, which has {nrows} rows x {ncols} "
                             f"columns (r0c0 ... {tile_name(nrows - 1, ncols - 1)}); the grid follows the "
                             f"latent size and spatial_split_param")
    rows = range(nrows) if r == "*" else [int(r)]
    cols = range(ncols) if c == "*" else [int(c)]
    return [(ri, cj) for ri in rows for cj in cols], (r != "*") + (c != "*")


def parse_tile_prompts(text, nrows, ncols):
    """'r0c1: sky', 'r1c*: crowd', 'r0c0, r0c2: wall', '*: rest' -> {(ri, cj): text}.

    A line without a key continues the prompt above it; '#' lines are comments.
    A more specific key wins (r0c1 over r0c* over *), then the later line. An
    empty prompt leaves the tile to the upscale's own conditioning, like a tile
    no key names.
    """
    entries = []
    for number, raw in enumerate(str(text).replace("\r\n", "\n").split("\n"), 1):
        if raw.lstrip().startswith("#"):
            continue
        m = LINE_RE.match(raw)
        if m:
            entries.append([[k.strip() for k in m.group(1).split(",")], [m.group(2).strip()]])
        elif raw.strip():
            if not entries:
                raise ValueError(f"tile prompts line {number}: start with a tile, e.g. 'r0c1: <prompt>' "
                                 f"(or '*: <prompt>' for every tile)")
            entries[-1][1].append(raw.strip())

    chosen = {}
    for order, (keys, lines) in enumerate(entries):
        prompt = "\n".join(line for line in lines if line).strip()
        for key in keys:
            tiles, priority = _key_tiles(key, nrows, ncols)
            for tile in tiles:
                if tile not in chosen or (priority, order) >= chosen[tile][:2]:
                    chosen[tile] = (priority, order, prompt)
    return {tile: prompt for tile, (_, _, prompt) in sorted(chosen.items()) if prompt}


def fill_template(template, prompt):
    """The template with {tile} replaced by the tile's prompt (appended when there is no {tile})."""
    template = str(template)
    if "{tile}" in template:
        return template.replace("{tile}", prompt).strip()
    return f"{template.strip()}\n{prompt}".strip()


def with_tile_text(cond, tile_cond):
    """The tile's encoded prompt, with everything else (keyframes, refs, anchors) taken from cond."""
    extras = {k: v for k, v in (cond[0][1] if cond else {}).items() if k not in TEXT_KEYS}
    return [[tensor, {**extras, **d}] for tensor, d in tile_cond]


def cell_at(starts, sizes, pos):
    """Index of the tile whose centre is nearest to pos (along one axis)."""
    return min(range(len(starts)), key=lambda i: abs(starts[i] + sizes[i] / 2 - pos))


def draw_tile_map(image, H, W, rows, cols, trows, tcols, prompts, max_side=1536):
    """[1, h, w, 3] picture of the grid: every tile outlined and labelled, green where it has its own prompt.

    image: a frame of the shot (any size: the grid is scaled to it), else a grey canvas.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if image is not None:
        frame = (image[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        canvas = Image.fromarray(frame[..., :3], "RGB")
    else:
        scale = min(1.0, max_side / (max(H, W) * 16))
        canvas = Image.new("RGB", (max(1, round(W * 16 * scale)), max(1, round(H * 16 * scale))), (64, 64, 64))
    sx, sy = canvas.width / W, canvas.height / H
    draw = ImageDraw.Draw(canvas)
    size = max(12, round(min(canvas.width, canvas.height) / 28))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    green, grey = (60, 230, 90), (235, 235, 235)
    for ri, (r0, tr) in enumerate(zip(rows, trows)):
        for cj, (c0, tc) in enumerate(zip(cols, tcols)):
            own = (ri, cj) in prompts
            inset = 2 * ((ri + cj) % 2)  # neighbours' outlines apart where tiles overlap
            box = [c0 * sx + inset, r0 * sy + inset, (c0 + tc) * sx - 1 - inset, (r0 + tr) * sy - 1 - inset]
            draw.rectangle(box, outline=green if own else grey, width=max(2, size // 8))
    for ri, (r0, tr) in enumerate(zip(rows, trows)):
        for cj, (c0, tc) in enumerate(zip(cols, tcols)):
            own = (ri, cj) in prompts
            label = tile_name(ri, cj) + ("" if own else " (main)")
            cx, cy = (c0 + tc / 2) * sx, (r0 + tr / 2) * sy
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            w, h = right - left, bottom - top
            pad = size // 4
            draw.rectangle([cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad],
                           fill=(20, 20, 20))
            draw.text((cx - w / 2 - left, cy - h / 2 - top), label, fill=green if own else grey, font=font)
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)[None]
