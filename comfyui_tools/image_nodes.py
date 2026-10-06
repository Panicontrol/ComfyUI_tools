"""Image utility nodes."""

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


def load_frame(path):
    """One file as ``([1, H, W, 3], [1, H, W])`` image and mask tensors."""
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
    fed in pieces. A batch needs one size for every frame, so frames that do
    not match the first one are an error unless you ask for them to be resized
    or skipped.
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
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("images", "masks", "count", "filenames")
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
        return str((folder, pattern, sort_by, reverse, start_index, step, count, on_size_mismatch, state))

    def load(self, directory, pattern, sort_by, reverse, start_index, step, count,
             on_size_mismatch):
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

        names = names[start_index::step]
        if count > 0:
            names = names[:count]
        if not names:
            raise ValueError(
                f"start_index {start_index} and step {step} skip past every file in {folder}"
            )

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

        return (torch.cat(frames), torch.cat(masks), len(loaded), "\n".join(loaded))


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


def write_mp4(path, pixels, frame_rate):
    """Encode ``[N, H, W, 3]`` uint8 frames as an H.264 mp4."""
    import av  # ships with ComfyUI
    from fractions import Fraction

    height, width = pixels.shape[1], pixels.shape[2]
    pad_h, pad_w = height % 2, width % 2  # yuv420p needs even sides
    with av.open(path, mode="w") as container:
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


class SaveImageSequence:
    """Save a render as video, plus an optional PNG sequence and zip of it.

    In <save_folder>, for new_folder = depth and version 4:
      depth_v004.mp4                        always
      depth_v004.zip                        need_zip: the frames, zipped
      depth_v004/depth_v004.01001.png ...   need_save: the frames

    number_version has ComfyUI's "control after generate" switch, set to
    increment, so every run is a new version. Masks go into the PNGs' alpha
    channel (1 - mask), which Load Image Sequence reads back as masks.
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
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("folder", "version", "count", "video")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = f"{CATEGORY}/image"
    DESCRIPTION = ("Save a render as <name>_v###.mp4 in the project folder, optionally with "
                   "its PNG sequence and a zip of it; a new version per run.")

    def save(self, images, save_folder, new_folder, number_version, start_frame_index,
             padding_frame_name, overwrite, masks=None, frame_rate=24.0, need_save=True,
             need_zip=False):
        import io
        import zipfile

        if Image is None:
            raise RuntimeError("this node needs Pillow and numpy (both ship with ComfyUI)")
        if not str(save_folder).strip():
            raise ValueError("give the node a save_folder")

        name = str(new_folder).strip()
        versioned = f"{name}_v{number_version:03d}" if name else f"v{number_version:03d}"
        project = output_directory(save_folder)
        folder = os.path.join(project, versioned)
        video_path = os.path.join(project, f"{versioned}.mp4")
        zip_path = os.path.join(project, f"{versioned}.zip")

        frames = int(images.shape[0])
        names = [
            f"{versioned}.{n:0{padding_frame_name}d}.png"
            for n in range(start_frame_index, start_frame_index + frames)
        ]

        targets = [video_path]
        if need_zip:
            targets.append(zip_path)
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
            alpha = (1.0 - resize_mask(masks, images.shape[2], images.shape[1])).clamp(0.0, 1.0)
            alpha = (alpha * 255.0).round().to(torch.uint8).cpu().numpy()

        os.makedirs(project, exist_ok=True)
        pixels = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()

        if need_save or need_zip:
            if need_save:
                os.makedirs(folder, exist_ok=True)
            archive = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) if need_zip else None
            try:
                for index, filename in enumerate(names):
                    if alpha is None:
                        image = Image.fromarray(pixels[index], "RGB")
                    else:
                        a = alpha[0 if alpha.shape[0] == 1 else index]
                        image = Image.fromarray(np.dstack([pixels[index], a]), "RGBA")
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG", compress_level=4)
                    data = buffer.getvalue()
                    if need_save:
                        with open(os.path.join(folder, filename), "wb") as handle:
                            handle.write(data)
                    if archive is not None:  # PNG is already compressed: store
                        archive.writestr(f"{versioned}/{filename}", data)
            finally:
                if archive is not None:
                    archive.close()

        write_mp4(video_path, pixels, float(frame_rate))

        return {"ui": {"text": [video_path]}, "result": (folder, number_version, frames, video_path)}

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
