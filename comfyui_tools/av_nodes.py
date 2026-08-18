"""Audio/video helper nodes for LanPaint-style AV inpainting workflows.

`LanPaint AV Encode` wants an ``audio_mask``: a MASK of shape ``[F]`` (or
``[F, 1]``) at the *video* frame rate, hard 0/1, where ``1`` means "regenerate
the audio at this moment" and ``0`` means "keep the original audio". When you
only inpaint the picture there is nothing upstream producing that mask, so
``AudioMask`` below builds one -- an all-zero stub by default.
"""

import json
import math

import torch

from .utils import CATEGORY


def parse_intervals(text):
    """Parse audio intervals in seconds into a list of ``(start, end)`` pairs.

    Accepts the JSON the LanPaint mask editor writes
    (``[{"start": 0.5, "end": 2.0}]``) as well as plain text ranges separated
    by newlines, commas or semicolons: ``0.5-2.0``, ``0.5:2``, ``0.5 2``.
    """
    text = (text or "").strip()
    if not text:
        return []

    if text[0] in "[{":
        try:
            data = json.loads(text)
        except ValueError as error:
            raise ValueError(f"audio intervals are not valid JSON: {error}") from error
        if isinstance(data, dict):
            data = [data]
        intervals = []
        for item in data:
            intervals.append((float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        return intervals

    intervals = []
    for chunk in text.replace(";", "\n").replace(",", "\n").splitlines():
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace("-", " ").replace(":", " ").split() if p]
        if len(parts) != 2:
            raise ValueError(f"expected a 'start-end' pair, got: {chunk!r}")
        intervals.append((float(parts[0]), float(parts[1])))
    return intervals


def video_frame_info(video):
    """Best-effort ``(frame_count, fps)`` for a ComfyUI ``VIDEO`` input."""
    frames = None
    fps = None

    getter = getattr(video, "get_frame_count", None)
    if getter is not None:
        try:
            frames = int(getter())
        except Exception:
            frames = None

    for name in ("get_fps", "get_frame_rate"):
        getter = getattr(video, name, None)
        if getter is None:
            continue
        try:
            fps = float(getter())
            break
        except Exception:
            fps = None

    if frames is None or fps is None:
        # falls back to decoding, which is why it is the last resort
        components = video.get_components()
        if frames is None:
            frames = int(components.images.shape[0])
        if fps is None:
            rate = getattr(components, "frame_rate", None)
            if rate is not None:
                fps = float(rate)

    return frames, fps


class AudioMask:
    """Build the ``audio_mask`` for LanPaint AV Encode.

    Use ``keep_all`` (the default) when you inpaint video only and want the
    original audio passed through untouched -- that is the stub AV Encode is
    missing. ``regenerate_all`` resamples the whole audio track, ``intervals``
    regenerates only the given time ranges.
    """

    MODES = ["keep_all", "regenerate_all", "intervals"]
    SHAPES = ["[F]", "[F, 1]"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {
                    "default": "keep_all",
                    "tooltip": "keep_all = all zeros (keep the original audio), "
                               "regenerate_all = all ones, intervals = only the listed seconds.",
                }),
                "intervals": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Seconds to regenerate, one range per line: '0.5-2.0'. "
                               "The editor's JSON ([{\"start\": 0.5, \"end\": 2.0}]) also works.",
                }),
                "frames": ("INT", {
                    "default": 81, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "Frame count used only when neither video nor video_mask is connected.",
                }),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                    "tooltip": "Frame rate used for intervals when no video is connected.",
                }),
                "invert": ("BOOLEAN", {"default": False}),
                "output_shape": (cls.SHAPES, {"default": "[F]"}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "Frame count and fps are read from the video."}),
                "video_mask": ("MASK", {"tooltip": "Per-frame video mask [F, H, W]; its frame count wins."}),
            },
        }

    RETURN_TYPES = ("MASK", "INT", "FLOAT")
    RETURN_NAMES = ("audio_mask", "frames", "fps")
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/av"
    DESCRIPTION = ("Audio mask [F] at video frame rate for LanPaint AV Encode "
                   "(1 = regenerate the audio, 0 = keep it). All zeros by default.")

    def resolve_length(self, frames, fps, video, video_mask):
        if video is not None:
            video_frames, video_fps = video_frame_info(video)
            if video_frames:
                frames = video_frames
            if video_fps:
                fps = video_fps
        if video_mask is not None:
            frames = int(video_mask.shape[0])
        return int(frames), float(fps)

    def build(self, mode, intervals, frames, fps, invert, output_shape,
              video=None, video_mask=None):
        frames, fps = self.resolve_length(frames, fps, video, video_mask)
        if frames < 1:
            raise ValueError("the audio mask needs at least one frame")

        if mode == "regenerate_all":
            mask = torch.ones(frames, dtype=torch.float32)
        else:
            mask = torch.zeros(frames, dtype=torch.float32)

        if mode == "intervals":
            for start, end in parse_intervals(intervals):
                if end <= start:
                    continue
                # same frame rounding as the LanPaint mask editor
                first = max(0, math.floor(start * fps))
                last = min(frames, math.ceil(end * fps))
                if last > first:
                    mask[first:last] = 1.0

        if invert:
            mask = 1.0 - mask
        if output_shape == "[F, 1]":
            mask = mask.unsqueeze(1)
        return (mask, frames, fps)


NODE_CLASS_MAPPINGS = {
    "ToolsAudioMask": AudioMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsAudioMask": "Audio Mask (tools)",
}
