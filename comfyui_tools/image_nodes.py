"""Image utility nodes."""

import json
import os
import re

import torch

from .utils import (
    CATEGORY,
    INTERPOLATION_MODES,
    hex_to_rgb,
    resize_image,
    resize_mask,
    round_to_multiple,
)

try:  # both ship with ComfyUI
    import numpy as np
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - exercised only without Pillow
    np = None
    Image = None
    ImageOps = None


def natural_key(name):
    """Sort key that orders frame_2 before frame_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def resolve_directory(directory):
    """Absolute path for a folder, allowing ~ and ComfyUI's input folder."""
    path = os.path.expanduser(str(directory).strip().strip('"'))
    if os.path.isabs(path):
        return path

    try:  # a relative path is most useful against ComfyUI's input folder
        import folder_paths

        candidate = os.path.join(folder_paths.get_input_directory(), path)
        if os.path.isdir(candidate):
            return candidate
    except Exception:
        pass
    return os.path.abspath(path)


def list_sequence(directory, pattern):
    """Files in ``directory`` matching any of the comma separated globs."""
    import fnmatch

    globs = [p.strip() for p in str(pattern).split(",") if p.strip()] or ["*"]
    names = []
    for name in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, name)):
            continue
        if any(fnmatch.fnmatch(name.lower(), g.lower()) for g in globs):
            names.append(name)
    return names


def png_bit_depth(path):
    """Bit depth from a PNG's header, 0 if the file is not a PNG."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(26)
    except OSError:
        return 0
    return head[24] if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) == 26 else 0


def load_png16(path):
    """A 16-bit PNG at full precision through OpenCV; None when OpenCV is not installed
    (Pillow then reads it at 8 bits)."""
    try:
        import cv2
    except ImportError:
        return None
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if data is None:
        return None
    data = data.astype(np.float32) / (65535.0 if data.dtype == np.uint16 else 255.0)
    if data.ndim == 2:
        data = data[..., None]
    if data.shape[-1] == 1:
        rgb, alpha = np.repeat(data, 3, -1), None
    elif data.shape[-1] == 2:
        rgb, alpha = np.repeat(data[..., :1], 3, -1), data[..., 1]
    else:
        rgb, alpha = data[..., 2::-1], (data[..., 3] if data.shape[-1] == 4 else None)
    frame = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
    if alpha is None:
        mask = torch.zeros((1, frame.shape[1], frame.shape[2]), dtype=torch.float32)
    else:
        mask = torch.from_numpy(1.0 - alpha).unsqueeze(0)
    return frame, mask


def load_frame(path):
    """One file as ``([1, H, W, 3], [1, H, W])`` image and mask tensors."""
    if png_bit_depth(path) == 16:
        loaded = load_png16(path)
        if loaded is not None:
            return loaded
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)

        if "A" in image.getbands():
            alpha = np.array(image.getchannel("A"), dtype=np.float32) / 255.0
            mask = torch.from_numpy(1.0 - alpha).unsqueeze(0)
        else:
            mask = None

        pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0

    frame = torch.from_numpy(pixels).unsqueeze(0)
    if mask is None:
        mask = torch.zeros((1, frame.shape[1], frame.shape[2]), dtype=torch.float32)
    return frame, mask


class ImageResize:
    """Resize an image by longest/shortest side, megapixels or explicit size."""

    MODES = ["longest_side", "shortest_side", "megapixels", "width_height"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "longest_side"}),
                "target_side": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 64.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "keep_aspect": ("BOOLEAN", {"default": True}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "interpolation": (INTERPOLATION_MODES, {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = f"{CATEGORY}/image"

    def target_size(self, src_w, src_h, mode, target_side, megapixels, width, height, keep_aspect):
        aspect = src_w / src_h
        if mode == "longest_side":
            if src_w >= src_h:
                return target_side, target_side / aspect
            return target_side * aspect, target_side
        if mode == "shortest_side":
            if src_w <= src_h:
                return target_side, target_side / aspect
            return target_side * aspect, target_side
        if mode == "megapixels":
            scale = ((megapixels * 1_000_000) / (src_w * src_h)) ** 0.5
            return src_w * scale, src_h * scale
        if not keep_aspect:
            return width, height
        scale = min(width / src_w, height / src_h)
        return src_w * scale, src_h * scale

    def resize(self, image, mode, target_side, megapixels, width, height,
               keep_aspect, multiple_of, interpolation):
        src_h, src_w = image.shape[1], image.shape[2]
        new_w, new_h = self.target_size(
            src_w, src_h, mode, target_side, megapixels, width, height, keep_aspect
        )
        new_w = round_to_multiple(new_w, multiple_of)
        new_h = round_to_multiple(new_h, multiple_of)
        return (resize_image(image, new_w, new_h, interpolation), new_w, new_h)


class ImagePadToRatio:
    """Pad an image to a target aspect ratio and return the padding as a mask."""

    POSITIONS = ["center", "top", "bottom", "left", "right"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ratio_width": ("FLOAT", {"default": 16.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "ratio_height": ("FLOAT", {"default": 9.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "position": (cls.POSITIONS, {"default": "center"}),
                "pad_color": ("STRING", {"default": "#000000"}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "pad_mask", "width", "height")
    FUNCTION = "pad"
    CATEGORY = f"{CATEGORY}/image"

    def offsets(self, position, extra_x, extra_y):
        left = extra_x // 2
        top = extra_y // 2
        if position == "left":
            left = 0
        elif position == "right":
            left = extra_x
        elif position == "top":
            top = 0
        elif position == "bottom":
            top = extra_y
        return left, top

    def pad(self, image, ratio_width, ratio_height, position, pad_color, multiple_of):
        batch, src_h, src_w, channels = image.shape
        ratio = ratio_width / ratio_height

        if src_w / src_h < ratio:
            new_w, new_h = src_h * ratio, float(src_h)
        else:
            new_w, new_h = float(src_w), src_w / ratio

        new_w = max(round_to_multiple(new_w, multiple_of), src_w)
        new_h = max(round_to_multiple(new_h, multiple_of), src_h)

        color = hex_to_rgb(pad_color)
        canvas = torch.empty(
            (batch, new_h, new_w, channels), dtype=image.dtype, device=image.device
        )
        for c in range(channels):
            canvas[..., c] = color[c] if c < len(color) else 1.0

        mask = torch.ones((batch, new_h, new_w), dtype=image.dtype, device=image.device)
        left, top = self.offsets(position, new_w - src_w, new_h - src_h)

        canvas[:, top:top + src_h, left:left + src_w, :] = image
        mask[:, top:top + src_h, left:left + src_w] = 0.0
        return (canvas, mask, new_w, new_h)


class ImageInfo:
    """Read the dimensions of an image batch as plain numbers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "batch_size", "aspect_ratio", "summary")
    FUNCTION = "info"
    CATEGORY = f"{CATEGORY}/image"

    def info(self, image):
        batch, height, width = image.shape[0], image.shape[1], image.shape[2]
        aspect = width / height
        summary = f"{width}x{height} | batch {batch} | aspect {aspect:.3f}"
        return (width, height, batch, aspect, summary)


class LoadImageSequence:
    """Load a folder of images -- a rendered PNG sequence -- as one IMAGE batch.

    The frames are ordered the way a sequence expects (``frame_2`` before
    ``frame_10``) and sliced with start/step/count, so a long render can be
    fed in pieces. ``index`` moves the window by ``count`` frames per run
    (control after generate: increment), so a graph queued once per index --
    by hand or by scripts/queue_sequence.py -- goes through a sequence too long
    to hold in memory, one piece at a time. A batch needs one size for every
    frame, so frames that do not match the first one are an error unless you
    ask for them to be resized or skipped.
    """

    SORT_MODES = ["name", "modified", "size"]
    MISMATCH = ["error", "resize", "skip"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "tooltip": "Folder holding the sequence. Absolute, ~ or relative to ComfyUI's input folder.",
                }),
                "pattern": ("STRING", {
                    "default": "*.png",
                    "tooltip": "Comma separated globs, e.g. '*.png' or 'frame_*.png, render_*.png'.",
                }),
                "sort_by": (cls.SORT_MODES, {
                    "default": "name",
                    "tooltip": "name sorts the way a sequence reads: frame_2 before frame_10.",
                }),
                "reverse": ("BOOLEAN", {"default": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "step": ("INT", {
                    "default": 1, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "Take every Nth frame.",
                }),
                "count": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "How many frames to load; 0 loads all of them.",
                }),
                "on_size_mismatch": (cls.MISMATCH, {
                    "default": "error",
                    "tooltip": "What to do with a frame that is not the size of the first one.",
                }),
            },
            "optional": {
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "control_after_generate": "increment",
                    "tooltip": "Which piece of count frames to load: frames start_index + index * count * step "
                               "on. With count 1, one frame per run; increment steps through the sequence.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "masks", "count", "filenames", "first_index", "frame_number", "total")
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/image"
    DESCRIPTION = "Load an image sequence from a folder as one batch, with the alpha channel as masks."

    @classmethod
    def sorted_names(cls, directory, pattern, sort_by, reverse):
        names = list_sequence(directory, pattern)
        if sort_by == "modified":
            names.sort(key=lambda name: os.path.getmtime(os.path.join(directory, name)))
        elif sort_by == "size":
            names.sort(key=lambda name: os.path.getsize(os.path.join(directory, name)))
        else:
            names.sort(key=natural_key)
        if reverse:
            names.reverse()
        return names

    @classmethod
    def IS_CHANGED(cls, directory, pattern, sort_by, reverse, start_index, step, count,
                   on_size_mismatch, **kwargs):
        # re-run when the folder's contents change, not just its path
        folder = resolve_directory(directory)
        if not os.path.isdir(folder):
            return f"missing:{folder}"
        state = [
            (name, os.path.getmtime(os.path.join(folder, name)), os.path.getsize(os.path.join(folder, name)))
            for name in sorted(list_sequence(folder, pattern))
        ]
        return str((folder, pattern, sort_by, reverse, start_index, step, count, on_size_mismatch,
                    kwargs.get("index", 0), state))

    def load(self, directory, pattern, sort_by, reverse, start_index, step, count,
             on_size_mismatch, index=0):
        if Image is None:
            raise RuntimeError("this node needs Pillow and numpy (both ship with ComfyUI)")

        folder = resolve_directory(directory)
        if not directory or not str(directory).strip():
            raise ValueError("give the node a folder to load the sequence from")
        if not os.path.isdir(folder):
            raise ValueError(f"not a folder: {folder}")

        names = self.sorted_names(folder, pattern, sort_by, reverse)
        if not names:
            raise ValueError(f"no files matching {pattern!r} in {folder}")

        positions = list(range(len(names)))[start_index::step]
        total = len(positions)
        if count > 0:
            positions = positions[index * count:(index + 1) * count]
        elif index > 0:
            raise ValueError("index needs a count: set count to the frames per run (1 for one frame)")
        if not positions:
            if index > 0:
                raise ValueError(
                    f"index {index} is past the end: {total} frames from start_index {start_index} with "
                    f"step {step} make {-(-total // count)} pieces of {count} (index 0 to {-(-total // count) - 1})"
                )
            raise ValueError(
                f"start_index {start_index} and step {step} skip past every file in {folder}"
            )
        first_index = positions[0]
        numbers = re.findall(r"\d+", os.path.splitext(names[first_index])[0])
        frame_number = int(numbers[-1]) if numbers else first_index
        names = [names[i] for i in positions]

        frames = []
        masks = []
        loaded = []
        width = height = None
        for name in names:
            frame, mask = load_frame(os.path.join(folder, name))
            if width is None:
                height, width = frame.shape[1], frame.shape[2]
            elif frame.shape[2] != width or frame.shape[1] != height:
                if on_size_mismatch == "skip":
                    continue
                if on_size_mismatch == "error":
                    raise ValueError(
                        f"{name} is {frame.shape[2]}x{frame.shape[1]}, but the batch is "
                        f"{width}x{height}; set on_size_mismatch to resize or skip"
                    )
                frame = resize_image(frame, width, height, "lanczos")
                mask = resize_mask(mask, width, height)
            frames.append(frame)
            masks.append(mask)
            loaded.append(name)

        if not frames:
            raise ValueError(f"every frame after the first was skipped as the wrong size in {folder}")

        return (torch.cat(frames), torch.cat(masks), len(loaded), "\n".join(loaded), first_index, frame_number,
                total)


def extend_indices(frames, length, mode, trim_longer=False):
    """Source frame for every output frame when stretching a sequence to ``length``.

    ``mirror`` plays the sequence forward, then backward, then forward again
    (0 1 2 1 0 1 ...) without repeating the frame it turns on, so the motion
    has no stutter at the turn. ``loop`` starts over (0 1 2 0 1 2 ...) and
    ``hold_last`` freezes on the final frame.
    """
    if frames < 1:
        raise ValueError("the sequence has no frames to extend")
    if length < 1:
        raise ValueError("the target length must be at least 1")

    if length <= frames:
        return list(range(length if trim_longer else frames))

    if mode == "loop":
        return [i % frames for i in range(length)]
    if mode == "hold_last":
        return list(range(frames)) + [frames - 1] * (length - frames)

    # mirror
    if frames == 1:
        return [0] * length
    period = 2 * frames - 2
    indices = []
    for i in range(length):
        k = i % period
        indices.append(k if k < frames else period - k)
    return indices


class ExtendSequence:
    """Stretch an image batch to a minimum length by mirroring or looping it.

    Built for models with a minimum clip length: a 98-frame sequence that has
    to be 124 frames plays forward, then mirrors back for the missing 26. A
    batch that is already long enough passes through untouched unless
    ``trim_longer`` cuts it to exactly ``length``.
    """

    MODES = ["mirror", "loop", "hold_last"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "length": ("INT", {
                    "default": 124, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "Frames the batch must have. Shorter batches are extended to it.",
                }),
                "mode": (cls.MODES, {
                    "default": "mirror",
                    "tooltip": "mirror: 0 1 2 1 0 1 ...  loop: 0 1 2 0 1 2 ...  hold_last: 0 1 2 2 2 ...",
                }),
                "trim_longer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: a batch already longer than length is kept whole. "
                               "On: it is cut to exactly length frames.",
                }),
            },
            "optional": {
                "masks": ("MASK", {"tooltip": "Extended with the same frame order as the images."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("images", "masks", "count", "frame_order")
    FUNCTION = "extend"
    CATEGORY = f"{CATEGORY}/image"
    DESCRIPTION = ("Extend an image batch to a minimum length by mirroring (ping-pong), "
                   "looping or holding the last frame, then trim to that length.")

    def extend(self, images, length, mode, trim_longer, masks=None):
        frames = int(images.shape[0])
        indices = extend_indices(frames, int(length), mode, trim_longer)
        order = torch.tensor(indices, dtype=torch.long, device=images.device)

        if masks is None:
            masks = torch.zeros(
                (frames, images.shape[1], images.shape[2]), dtype=torch.float32, device=images.device
            )
        elif masks.ndim == 2:  # a single [H, W] mask
            masks = masks.unsqueeze(0)

        if masks.shape[0] == 1 and frames > 1:
            masks = masks.expand(frames, -1, -1)
        elif masks.shape[0] != frames:
            raise ValueError(
                f"the masks have {masks.shape[0]} frames but the images have {frames}"
            )

        out_images = images.index_select(0, order)
        out_masks = masks.index_select(0, order.to(masks.device)).contiguous()
        return (out_images, out_masks, len(indices), ",".join(str(i) for i in indices))


class RestoreSequenceLength:
    """Cut a generated batch back to the length of the sequence it came from.

    The counterpart of Extend Sequence: every one of its modes keeps the
    original frames first and appends the padding after them, so the first N
    frames of the generated batch are the original timeline. Connect the
    pre-extension batch as ``original`` and N is its frame count; otherwise
    ``length`` sets it. A batch that is already that short or shorter passes
    through unchanged -- no frames are invented.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The generated (extended) batch."}),
                "length": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Frames to keep. Ignored when original is connected.",
                }),
            },
            "optional": {
                "original": ("IMAGE", {
                    "tooltip": "The batch before Extend Sequence; its frame count is the length to restore.",
                }),
                "masks": ("MASK", {"tooltip": "Cut to the same length as the images."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("images", "masks", "count")
    FUNCTION = "restore"
    CATEGORY = f"{CATEGORY}/image"
    DESCRIPTION = ("Trim a generated batch back to the original sequence length, "
                   "undoing Extend Sequence before saving.")

    def restore(self, images, length, original=None, masks=None):
        target = int(original.shape[0]) if original is not None else int(length)
        if target < 1:
            raise ValueError("connect the original batch or set length to the frames to keep")

        frames = int(images.shape[0])
        keep = min(target, frames)
        out_images = images[:keep]

        if masks is None:
            out_masks = torch.zeros(
                (keep, images.shape[1], images.shape[2]), dtype=torch.float32, device=images.device
            )
        else:
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            if masks.shape[0] == 1:
                out_masks = masks.expand(keep, -1, -1).contiguous()
            elif masks.shape[0] >= keep:
                out_masks = masks[:keep]
            else:
                raise ValueError(
                    f"the masks have {masks.shape[0]} frames, fewer than the {keep} being kept"
                )

        return (out_images, out_masks, keep)


def output_directory(directory):
    """Absolute folder for saving: absolute, ~, or relative to ComfyUI's output folder."""
    path = os.path.expanduser(str(directory).strip().strip('"'))
    if os.path.isabs(path):
        return path
    try:
        import folder_paths

        return os.path.join(folder_paths.get_output_directory(), path)
    except Exception:
        return os.path.abspath(path)


def png16_bytes(array):
    """PNG file bytes of an ``[H, W, 3 | 4]`` uint16 array, 16 bits per channel.

    Pillow cannot write 16-bit RGB, so the file is built here: Sub filter on
    every row, zlib, nothing else needed by any PNG reader.
    """
    import struct
    import zlib

    height, width, channels = array.shape
    rows = np.ascontiguousarray(array.astype(">u2")).view(np.uint8).reshape(height, width * channels * 2)
    bpp = channels * 2
    filtered = rows.copy()
    filtered[:, bpp:] = rows[:, bpp:] - rows[:, :-bpp]  # uint8 arithmetic wraps mod 256, as PNG wants
    body = np.concatenate([np.ones((height, 1), np.uint8), filtered], 1).tobytes()

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 16, 2 if channels == 3 else 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(body, 4))
            + chunk(b"IEND", b""))


FRAME_FORMATS = {"png": ".png", "jpg": ".jpg", "tiff": ".tif", "exr": ".exr"}
EXR_COMPRESSIONS = ["piz", "dwaa", "zip"]
VIDEO_OVERSIZE = ["downscale", "skip", "error"]
X264_MAX_SIDE = 16384  # libx264 refuses wider or taller frames


def srgb_to_linear(rgb):
    """Inverse sRGB transfer (IEC 61966-2-1), as ComfyUI's Save Image Advanced does for EXR."""
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)


def codec_bytes(codec_name, array, frame_fmt, options):
    """One frame through an FFmpeg image encoder (PyAV); the encoder's output is the file."""
    import av
    from fractions import Fraction

    codec = av.CodecContext.create(codec_name, "w")
    codec.height, codec.width = array.shape[0], array.shape[1]
    codec.pix_fmt = frame_fmt
    codec.time_base = Fraction(1, 1)
    codec.options = options
    frame = av.VideoFrame.from_ndarray(array, format=frame_fmt)
    frame.pts = 0
    return b"".join(bytes(p) for p in list(codec.encode(frame)) + list(codec.encode(None)))


def exr_bytes(rgb, alpha, compression):
    """Half-float EXR of linear ``rgb`` [H, W, 3] (and ``alpha``): PIZ or DWAA through the
    OpenEXR package, ZIP through FFmpeg (as ComfyUI's own Save Image Advanced) without it."""
    channels = rgb if alpha is None else torch.cat([rgb, alpha[..., None]], -1)
    data = channels.to(torch.float16).cpu().numpy()
    try:
        import OpenEXR
    except ImportError:
        OpenEXR = None
        if compression != "zip":
            import logging

            logging.warning("[Save Image Sequence] the OpenEXR package is not installed (pip install OpenEXR): "
                            "writing %s EXRs with ZIP instead", compression.upper())
    if OpenEXR is None or compression == "zip":
        fmt = "gbrpf32le" if alpha is None else "gbrapf32le"  # PyAV takes RGB(A) and plans it itself
        frame = np.ascontiguousarray(data.astype(np.float32))
        return codec_bytes("exr", frame, fmt, {"format": "half", "compression": "zip16"})
    import tempfile

    header = {"compression": getattr(OpenEXR, f"{compression.upper()}_COMPRESSION"), "type": OpenEXR.scanlineimage}
    handle, path = tempfile.mkstemp(suffix=".exr")
    os.close(handle)
    try:
        with OpenEXR.File(header, {"RGB" if alpha is None else "RGBA": data}) as exr:
            exr.write(path)
        with open(path, "rb") as stored:
            return stored.read()
    finally:
        os.remove(path)


def encode_frame(frame, alpha, file_format, bit_depth, jpg_quality=95, exr_compression="piz", noise=None):
    """File bytes of one ``[H, W, 3]`` float frame (``alpha`` [H, W] or None) in the given format.

    8-bit outputs are rounded once (after ``noise``, a dither pattern in 0-255 units, if any);
    16-bit ones go from float straight to uint16; EXR gets linear light from the sRGB frame.
    """
    import io

    frame = frame[..., :3].float()
    if file_format == "exr":
        return exr_bytes(srgb_to_linear(frame), None if alpha is None else alpha.float(), exr_compression)
    if str(bit_depth) == "16":
        if alpha is not None:
            frame = torch.cat([frame, alpha.float()[..., None]], -1)
        data = (frame.clamp(0.0, 1.0) * 65535.0).round().to(torch.int32).cpu().numpy().astype(np.uint16)
        if file_format == "tiff":
            fmt = "rgb48le" if data.shape[-1] == 3 else "rgba64le"
            return codec_bytes("tiff", np.ascontiguousarray(data), fmt, {"compression_algo": "deflate"})
        return png16_bytes(data)
    values = frame * 255.0
    if noise is not None:
        values = values + noise
    pixels = values.clamp(0.0, 255.0).round().to(torch.uint8).cpu().numpy()
    if alpha is not None and file_format != "jpg":
        a = (alpha.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        image = Image.fromarray(np.dstack([pixels, a]), "RGBA")
    else:
        image = Image.fromarray(pixels, "RGB")
    buffer = io.BytesIO()
    if file_format == "jpg":
        image.save(buffer, format="JPEG", quality=int(jpg_quality), subsampling=0)  # 4:4:4
    elif file_format == "tiff":
        image.save(buffer, format="TIFF", compression="tiff_adobe_deflate")
    else:
        image.save(buffer, format="PNG", compress_level=4)
    return buffer.getvalue()


def write_file(path, data):
    """Write through a temporary name, so a crash never leaves a half-written frame under its real name."""
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, path)


# background frame writing (async_write): a few threads, a bound on the frames they hold
_WRITER = None
_PENDING = []
_IN_FLIGHT = None


def _writer():
    global _WRITER, _IN_FLIGHT
    if _WRITER is None:
        import atexit
        import threading
        from concurrent.futures import ThreadPoolExecutor

        _WRITER = ThreadPoolExecutor(max_workers=3, thread_name_prefix="tools-save")
        _IN_FLIGHT = threading.BoundedSemaphore(4)  # 4 large frames at most wait in memory
        atexit.register(wait_for_writes)
    return _WRITER


def submit_write(job):
    """Run ``job`` on the background writer; at most a few frames are held at a time."""
    _writer()
    _IN_FLIGHT.acquire()
    future = _WRITER.submit(job)
    future.add_done_callback(lambda _: _IN_FLIGHT.release())
    _PENDING.append(future)
    return future


def wait_for_writes(futures=None):
    """Wait for background writes (all of them, or ``futures``) and raise the first failure."""
    waiting = list(_PENDING if futures is None else futures)
    errors = []
    for future in waiting:
        try:
            future.result()
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(error)
        if future in _PENDING:
            _PENDING.remove(future)
    if errors:
        raise RuntimeError(f"a frame written in the background failed: {errors[0]}") from errors[0]


def finished_write_errors():
    """Failures of background writes that have finished since the last call (pending ones stay)."""
    done = [future for future in _PENDING if future.done()]
    wait_for_writes(done)


def preview_frames(images, max_side):
    """uint8 frames scaled so the longer side is ``max_side`` (even sides), for an mp4 preview."""
    import torch.nn.functional as F

    height, width = images.shape[1], images.shape[2]
    scale = max_side / max(width, height)
    new_w = max(2, int(round(width * scale / 2)) * 2)
    new_h = max(2, int(round(height * scale / 2)) * 2)
    frames = []
    for frame in images:
        small = F.interpolate(frame[..., :3].float().movedim(-1, 0)[None], size=(new_h, new_w), mode="area")
        frames.append((small[0].movedim(0, -1).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy())
    return np.stack(frames)


def write_mp4(path, pixels, frame_rate, metadata=None):
    """Encode ``[N, H, W, 3]`` uint8 frames as an H.264 mp4.

    ``metadata`` values are written as container tags (dicts as JSON), the way
    ComfyUI's own Save Video stores ``workflow`` and ``prompt`` -- dropping the
    mp4 onto ComfyUI then loads the workflow back.
    """
    import av  # ships with ComfyUI
    from fractions import Fraction

    height, width = pixels.shape[1], pixels.shape[2]
    pad_h, pad_w = height % 2, width % 2  # yuv420p needs even sides
    # without use_metadata_tags the mp4 muxer drops custom keys
    options = {"movflags": "use_metadata_tags+faststart"}
    with av.open(path, mode="w", options=options) as container:
        for key, value in (metadata or {}).items():
            container.metadata[key] = value if isinstance(value, str) else json.dumps(value)
        stream = container.add_stream("h264", rate=Fraction(round(frame_rate * 1000), 1000))
        stream.width = width + pad_w
        stream.height = height + pad_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for frame in pixels:
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def workflow_metadata(prompt, extra_pnginfo):
    """ComfyUI's workflow/prompt tags, unless metadata is disabled (--disable-metadata)."""
    try:
        from comfy.cli_args import args

        if args.disable_metadata:
            return None
    except Exception:
        pass
    metadata = dict(extra_pnginfo or {})  # carries "workflow"
    if prompt is not None:
        metadata["prompt"] = prompt
    return metadata or None


class SaveImageSequence:
    """Save a render as a frame sequence and a video.

    In <save_folder>, for new_folder = depth and version 4:
      depth_v004/depth_v004.01001.png ...   need_save: the frames (png, jpg, tiff or exr)
      depth_v004.mp4                        need_video: H.264, a smaller preview when too large
      depth_v004.latent                     latent connected: for Load Latent From
      depth_v004.notes.txt                  notes connected: "depth_v004" + the notes
      depth_v004.zip                        need_zip: the frames, zipped

    The frames are written first (each under a temporary name until complete),
    the mp4 after them, so a failing mp4 never costs frames. number_version has
    ComfyUI's "control after generate" switch, set to increment, so every run
    is a new version. Masks go into the frames' alpha channel (1 - mask), which
    Load Image Sequence reads back as masks. Notes also go into the mp4 as its
    'notes' tag.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "save_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Project folder: the video, zip and sequence folder are written here.",
                }),
                "new_folder": ("STRING", {
                    "default": "render",
                    "tooltip": "Render name: gives depth_v004.mp4 and depth_v004/depth_v004.01001.png.",
                }),
                "number_version": ("INT", {
                    "default": 1, "min": 0, "max": 999, "step": 1,
                    "control_after_generate": "increment",
                }),
                "start_frame_index": ("INT", {"default": 1001, "min": 0, "max": 10000000, "step": 1}),
                "padding_frame_name": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: stop with an error instead of replacing files that already exist.",
                }),
            },
            # appended after the original widgets so saved workflows keep their
            # values in place; optional so those workflows still validate
            "optional": {
                "masks": ("MASK", {"tooltip": "Saved as the PNG alpha channel."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "need_save": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also save the PNG sequence in <save_folder>/<name>_v###/.",
                }),
                "need_zip": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also zip the PNG sequence next to the video as <name>_v###.zip.",
                }),
                "embed_workflow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write the workflow into the mp4, so dropping it onto ComfyUI loads the graph.",
                }),
                "save_workflow_json": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also save the workflow next to the video as <name>_v###.json.",
                }),
                "latent": ("LATENT", {
                    "tooltip": "Also save this latent next to the video as <name>_v###.latent "
                               "(MiniMax H3 video+audio included); load it with Load Latent From.",
                }),
                "notes": ("STRING", {
                    "forceInput": True,
                    "tooltip": "What this version is made of, e.g. the info of the graph's Shot Paths joined "
                               "with Text Concat. Saved under the version name as the mp4's notes tag and "
                               "as <name>_v###.notes.txt; empty = neither.",
                }),
                "bit_depth": (["8", "16"], {
                    "default": "8",
                    "tooltip": "Bits per channel of the PNGs. 16 keeps float results (e.g. H3 DeGrid's) "
                               "without 8-bit rounding steps; the mp4 stays 8-bit.",
                }),
                "dither": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Triangular dither (about 0.4 of an 8-bit step) before rounding to 8 bits, for "
                               "8-bit PNGs and the mp4: after H3 DeGrid, plain rounding puts a step back on many "
                               "block edges. One pattern for all frames, the same on every run; it still adds a "
                               "little flicker, so keep PNGs at 16 bits where you can.",
                }),
                "need_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also write <name>_v###.mp4 after the frames. Off for per-frame runs.",
                }),
                "video_max_side": ("INT", {
                    "default": 8192, "min": 64, "max": X264_MAX_SIDE, "step": 2,
                    "tooltip": "Frames wider or taller than this follow video_oversize (libx264 stops at 16384).",
                }),
                "video_oversize": (VIDEO_OVERSIZE, {
                    "default": "downscale",
                    "tooltip": "Frames over video_max_side: downscale = an mp4 preview with that longer side; "
                               "skip = no mp4 (a warning); error = stop after the frames are written.",
                }),
                "format": (list(FRAME_FORMATS), {
                    "default": "png",
                    "tooltip": "Frame files. jpg is 8-bit 4:4:4; tiff is deflate-compressed, 8 or 16 bits; "
                               "exr is half float, linear light from the sRGB frames.",
                }),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "exr_compression": (EXR_COMPRESSIONS, {
                    "default": "piz",
                    "tooltip": "piz (lossless) and dwaa (lossy, small) need the OpenEXR package (pip install "
                               "OpenEXR); without it, and with zip, the EXR is ZIP-compressed (lossless).",
                }),
                "async_write": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Encode and write frames on background threads and return at once, so the GPU "
                               "goes on with the next run. Waited for before the mp4 or zip; a failure shows on "
                               "the next run.",
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("folder", "version", "count", "video", "latent_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = f"{CATEGORY}/image"
    DESCRIPTION = ("Save a render as a frame sequence (png, jpg, tiff, exr) and <name>_v###.mp4 in the "
                   "project folder; a new version per run.")

    @staticmethod
    def format_problem(file_format, bit_depth):
        if file_format not in FRAME_FORMATS:
            return f"format {file_format!r} is not one of {', '.join(FRAME_FORMATS)}"
        if str(bit_depth) not in ("8", "16"):
            return f"bit_depth {bit_depth!r} is not 8 or 16"
        if file_format == "jpg" and str(bit_depth) == "16":
            return "jpg is 8-bit only: use png or tiff for 16 bits, or exr"
        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, format="png", bit_depth="8"):
        return cls.format_problem(format, bit_depth) or True

    def save(self, images, save_folder, new_folder, number_version, start_frame_index,
             padding_frame_name, overwrite, masks=None, frame_rate=24.0, need_save=True,
             need_zip=False, embed_workflow=True, save_workflow_json=False, latent=None, notes=None,
             bit_depth="8", dither=False, need_video=True, video_max_side=8192, video_oversize="downscale",
             format="png", jpg_quality=95, exr_compression="piz", async_write=False, prompt=None,
             extra_pnginfo=None):
        import logging
        import zipfile

        if Image is None:
            raise RuntimeError("this node needs Pillow and numpy (both ship with ComfyUI)")
        if not str(save_folder).strip():
            raise ValueError("give the node a save_folder")
        file_format = str(format)
        problem = self.format_problem(file_format, bit_depth)
        if problem:
            raise ValueError(problem)
        finished_write_errors()  # a background write of an earlier run that failed shows here

        name = str(new_folder).strip()
        versioned = f"{name}_v{number_version:03d}" if name else f"v{number_version:03d}"
        project = output_directory(save_folder)
        folder = os.path.join(project, versioned)
        video_path = os.path.join(project, f"{versioned}.mp4")
        zip_path = os.path.join(project, f"{versioned}.zip")
        json_path = os.path.join(project, f"{versioned}.json")
        latent_path = os.path.join(project, f"{versioned}.latent") if latent is not None else ""
        # the version name first, so the text says which file it belongs to
        notes_text = f"{versioned}\n{notes}" if notes is not None and str(notes).strip() else ""
        notes_path = os.path.join(project, f"{versioned}.notes.txt") if notes_text else ""

        frames, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        extension = FRAME_FORMATS[file_format]
        names = [
            f"{versioned}.{n:0{padding_frame_name}d}{extension}"
            for n in range(start_frame_index, start_frame_index + frames)
        ]

        video = "none"
        if need_video:
            video = "full"
            if max(width, height) > int(video_max_side):
                video = {"downscale": "preview", "skip": "none", "error": "error"}[video_oversize]
                if video == "none":
                    logging.warning("[Save Image Sequence] %dx%d is over video_max_side %d: no mp4 (video_oversize "
                                    "skip)", width, height, video_max_side)

        targets = [video_path] if video in ("full", "preview") else []
        if need_zip:
            targets.append(zip_path)
        if save_workflow_json:
            targets.append(json_path)
        if latent_path:
            targets.append(latent_path)
        if notes_path:
            targets.append(notes_path)
        if need_save:
            targets += [os.path.join(folder, n) for n in names]
        if not overwrite:
            existing = [path for path in targets if os.path.exists(path)]
            if existing:
                raise ValueError(
                    f"{os.path.basename(existing[0])} already exists in {os.path.dirname(existing[0])}; "
                    "raise number_version or turn overwrite on"
                )

        alpha = None
        if masks is not None:
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            if masks.shape[0] not in (1, frames):
                raise ValueError(f"the masks have {masks.shape[0]} frames but the images have {frames}")
            if file_format == "jpg":
                logging.warning("[Save Image Sequence] jpg has no alpha channel: the masks are not saved")
            else:
                alpha = (1.0 - resize_mask(masks, width, height)).clamp(0.0, 1.0)
        noise = None
        if dither:  # TPDF, one pattern for every frame (a new one per frame flickers), same on a rerun
            g = torch.Generator().manual_seed(int(number_version))
            noise = torch.rand((height, width, 3), generator=g) - torch.rand((height, width, 3), generator=g)

        os.makedirs(project, exist_ok=True)
        if need_save:
            os.makedirs(folder, exist_ok=True)

        # 1. the frames
        def frame_job(index, filename):
            def job():
                a = None if alpha is None else alpha[0 if alpha.shape[0] == 1 else index]
                data = encode_frame(images[index], a, file_format, bit_depth, jpg_quality, exr_compression, noise)
                if need_save:
                    write_file(os.path.join(folder, filename), data)
                return data if need_zip else None
            return job

        jobs = [frame_job(i, n) for i, n in enumerate(names)] if need_save or need_zip else []
        if async_write:
            futures = [submit_write(job) for job in jobs]
            encoded = None
        else:
            futures, encoded = [], [job() for job in jobs]

        if need_zip:
            if futures:
                wait_for_writes(futures)
                encoded = [future.result() for future in futures]
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:  # frames are compressed: store
                for filename, data in zip(names, encoded):
                    archive.writestr(f"{versioned}/{filename}", data)

        # 2. the video, after the frames
        written_video = ""
        if video in ("full", "preview"):
            if futures:
                wait_for_writes(futures)
            metadata = workflow_metadata(prompt, extra_pnginfo) if embed_workflow else None
            if notes_text:
                metadata = {**(metadata or {}), "notes": notes_text}
            if video == "preview":
                pixels = preview_frames(images, int(video_max_side))
            else:
                values = images[..., :3].float() * 255.0
                if noise is not None:
                    values = values + noise
                pixels = values.clamp(0.0, 255.0).round().to(torch.uint8).cpu().numpy()
            try:
                write_mp4(video_path, pixels, float(frame_rate), metadata)
            except Exception as error:
                raise RuntimeError(f"the frames are saved in {folder}, but the mp4 failed: {error}") from error
            written_video = video_path

        if notes_path:
            with open(notes_path, "w", encoding="utf-8") as handle:
                handle.write(notes_text + "\n")

        if save_workflow_json:
            # the UI graph loads back by dropping the file on ComfyUI; a run
            # queued through the API has none, so fall back to its prompt
            workflow = (extra_pnginfo or {}).get("workflow") or prompt or {}
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(workflow, handle, indent=2, ensure_ascii=False)

        if latent_path:
            from .latent_nodes import write_latent

            write_latent(latent_path, latent)

        if video == "error":
            raise ValueError(
                f"frame {width}x{height}: {max(width, height)} px is over video_max_side {video_max_side} "
                f"(libx264 takes at most {X264_MAX_SIDE} px); the frames are saved in {folder}. Set "
                f"video_oversize to downscale or skip, or turn need_video off"
            )
        shown = written_video or (folder if need_save else project)
        return {"ui": {"text": [shown]}, "result": (folder, number_version, frames, written_video, latent_path)}

NODE_CLASS_MAPPINGS = {
    "ToolsImageResize": ImageResize,
    "ToolsImagePadToRatio": ImagePadToRatio,
    "ToolsImageInfo": ImageInfo,
    "ToolsLoadImageSequence": LoadImageSequence,
    "ToolsExtendSequence": ExtendSequence,
    "ToolsRestoreSequenceLength": RestoreSequenceLength,
    "ToolsSaveImageSequence": SaveImageSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsImageResize": "Image Resize (tools)",
    "ToolsImagePadToRatio": "Image Pad To Ratio (tools)",
    "ToolsImageInfo": "Image Info (tools)",
    "ToolsLoadImageSequence": "Load Image Sequence (tools)",
    "ToolsExtendSequence": "Extend Sequence (tools)",
    "ToolsRestoreSequenceLength": "Restore Sequence Length (tools)",
    "ToolsSaveImageSequence": "Save Image Sequence (tools)",
}
