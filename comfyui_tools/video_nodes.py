"""Video file nodes."""

import os
from fractions import Fraction

from .utils import CATEGORY

try:  # ships with ComfyUI
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - exercised only without Pillow
    Image = ImageDraw = ImageFont = None

STAMP_FONTS = (
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "DejaVuSansMono-Bold.ttf",  # Pillow finds it in the system font folders
)


def parse_reel_items(text):
    """Blocks separated by blank lines: a clip path, then its stamp lines."""
    clips, block = [], []
    for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n") + [""]:
        if line.strip():
            block.append(line.rstrip())
            continue
        if block:
            clips.append((block[0].strip().strip('"\''), block[1:]))
            block = []
    return clips


def stamp_font(size):
    for name in STAMP_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1 has a single fixed-size default font
        return ImageFont.load_default()


def stamp_tile(lines, font, font_size, box_opacity):
    """The stamp as an RGBA tile: text on a translucent black box."""
    pad, gap = font_size // 2, font_size // 3
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    width = max(int(font.getlength(line)) for line in lines) + 2 * pad
    height = len(lines) * line_height + (len(lines) - 1) * gap + 2 * pad
    tile = Image.new("RGBA", (width, height), (0, 0, 0, round(255 * box_opacity)))
    draw = ImageDraw.Draw(tile)
    for index, line in enumerate(lines):
        draw.text((pad, pad + index * (line_height + gap)), line, font=font, fill=(255, 255, 255, 255))
    return tile


def fit_frame(image, width, height):
    """Scale to fit width x height keeping the aspect ratio, centred on black."""
    if image.size == (width, height):
        return image
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = Image.new("RGB", (width, height))
    canvas.paste(image.resize(size, Image.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


class VideoReel:
    """Concatenate mp4 clips into one reel, stamping each with its own text.

    ``items`` holds one block per clip, blocks separated by a blank line: the
    first line is the clip's absolute path, the rest are its stamp lines.
    Frames are streamed one at a time, so long reels never sit in memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One block per clip, separated by a blank line: the mp4 path, "
                               "then the stamp text, one line per line.",
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path of the reel .mp4; its folder is created if missing.",
                }),
                "width": ("INT", {"default": 1920, "min": 2, "max": 16384, "step": 2}),
                "height": ("INT", {"default": 1088, "min": 2, "max": 16384, "step": 2}),
                "frame_step": ("INT", {
                    "default": 1, "min": 1, "max": 12, "step": 1,
                    "tooltip": "Keep every Nth frame of each clip: 2 keeps every second one.",
                }),
                "frame_rate": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "fps of the reel: 12 for frame_step 2 on 24 fps clips.",
                }),
                "font_size": ("INT", {"default": 28, "min": 6, "max": 512, "step": 1}),
                "box_opacity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "draw_stamp": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: stop with an error instead of replacing an existing file.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("video", "frames")
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = f"{CATEGORY}/video"
    DESCRIPTION = ("Concatenate mp4 clips into one reel with a text stamp on each clip, "
                   "optionally keeping every Nth frame.")

    @classmethod
    def IS_CHANGED(cls, items, output_path, **kwargs):
        # rebuild when a clip is re-rendered or the reel was deleted
        stats = []
        for path, _ in parse_reel_items(items):
            try:
                stat = os.stat(path)
                stats.append((path, stat.st_mtime, stat.st_size))
            except OSError:
                stats.append((path, None))
        return repr((stats, os.path.exists(str(output_path).strip().strip('"\''))))

    def build(self, items, output_path, width, height, frame_step, frame_rate, font_size,
              box_opacity, crf, draw_stamp, overwrite):
        import av  # ships with ComfyUI

        if Image is None:
            raise RuntimeError("this node needs Pillow (it ships with ComfyUI)")
        clips = parse_reel_items(items)
        if not clips:
            raise ValueError("items is empty: give each clip a block starting with its mp4 path")
        missing = [path for path, _ in clips if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("clips not found:\n" + "\n".join(missing))

        output_path = str(output_path).strip().strip('"\'')
        if not output_path:
            raise ValueError("give the node an output_path")
        if os.path.exists(output_path) and not overwrite:
            raise ValueError(f"{output_path} already exists; turn overwrite on or change output_path")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        width, height = width + width % 2, height + height % 2  # yuv420p needs even sides
        font = stamp_font(font_size) if draw_stamp else None
        margin = font_size
        written = 0
        try:
            with av.open(output_path, mode="w", options={"movflags": "faststart"}) as container:
                stream = container.add_stream("h264", rate=Fraction(round(frame_rate * 1000), 1000))
                stream.width, stream.height = width, height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": str(crf), "preset": "medium"}
                for path, lines in clips:
                    tile = stamp_tile(lines, font, font_size, box_opacity) if draw_stamp and lines else None
                    try:
                        with av.open(path) as source:
                            for index, frame in enumerate(source.decode(video=0)):
                                if index % frame_step:
                                    continue
                                image = fit_frame(frame.to_image(), width, height)
                                if tile is not None:
                                    image.paste(tile, (margin, margin), tile)
                                for packet in stream.encode(av.VideoFrame.from_image(image)):
                                    container.mux(packet)
                                written += 1
                    except av.FFmpegError as error:
                        raise RuntimeError(f"cannot decode {path}: {error}") from error
                if not written:
                    raise ValueError("the clips have no video frames")
                for packet in stream.encode():
                    container.mux(packet)
        except BaseException:
            try:  # do not leave a broken reel behind
                os.remove(output_path)
            except OSError:
                pass
            raise

        return {"ui": {"text": [f"{output_path} ({written} frames)"]}, "result": (output_path, written)}


NODE_CLASS_MAPPINGS = {
    "ToolsVideoReel": VideoReel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsVideoReel": "Tools Video Reel: concat + stamp",
}
