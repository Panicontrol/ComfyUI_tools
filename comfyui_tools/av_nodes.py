"""Audio/video helper nodes for LanPaint-style AV inpainting workflows.

`LanPaint AV Encode` wants an ``audio_mask``: a MASK of shape ``[F]`` (or
``[F, 1]``) at the *video* frame rate, hard 0/1, where ``1`` means "regenerate
the audio at this moment" and ``0`` means "keep the original audio". When you
only inpaint the picture there is nothing upstream producing that mask, so
``AudioMask`` below builds one -- an all-zero stub by default.

AV Encode also reads the audio track from the VIDEO itself, so a clip with no
sound at all fails before the mask is even used ("the video has no audio track
to encode"). ``SilentAudio`` and ``VideoAddSilentAudio`` fill that gap with a
silent track of the right length.
"""

import json
import math
from fractions import Fraction

import torch

from .utils import CATEGORY

try:  # available when running inside ComfyUI
    from comfy_api.latest._input_impl.video_types import VideoFromComponents
    from comfy_api.latest._util.video_types import VideoComponents
except Exception:  # pragma: no cover - exercised only outside ComfyUI
    VideoFromComponents = None
    VideoComponents = None

CHANNEL_LAYOUTS = {"mono": 1, "stereo": 2}


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


def silent_audio(seconds, sample_rate, channels="stereo", batch_size=1):
    """Build a ComfyUI ``AUDIO`` dict holding ``seconds`` of silence."""
    if seconds <= 0:
        raise ValueError("the silence needs a positive duration")
    if sample_rate <= 0:
        raise ValueError("the sample rate must be positive")

    count = CHANNEL_LAYOUTS.get(channels, channels if isinstance(channels, int) else 2)
    samples = int(math.ceil(seconds * sample_rate))
    waveform = torch.zeros((int(batch_size), int(count), samples), dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def video_seconds(video):
    """Duration of a ``VIDEO`` in seconds, from its frame count and frame rate."""
    frames, fps = video_frame_info(video)
    if not frames or not fps:
        raise ValueError("could not read the frame count and frame rate of the video")
    return frames / fps


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


class SilentAudio:
    """Generate a silent ``AUDIO`` track.

    Handy whenever a node insists on audio you do not have -- the MiniMax H3
    audio VAE runs at 32 kHz, so the default sample rate avoids a resample.
    """

    CHANNELS = ["stereo", "mono"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.01, "max": 3600.0, "step": 0.01,
                    "tooltip": "Duration, used when no video is connected and frames is 0.",
                }),
                "frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Length in video frames; 0 means use the seconds widget.",
                }),
                "fps": ("FLOAT", {"default": 25.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "sample_rate": ("INT", {
                    "default": 32000, "min": 1000, "max": 192000, "step": 100,
                    "tooltip": "32000 matches the MiniMax H3 audio VAE, so nothing has to be resampled.",
                }),
                "channels": (cls.CHANNELS, {"default": "stereo"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "The duration is taken from the video when connected."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT")
    RETURN_NAMES = ("audio", "seconds", "samples")
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY}/av"
    DESCRIPTION = "Silent audio track of a given length (from a video, a frame count or seconds)."

    def generate(self, seconds, frames, fps, sample_rate, channels, batch_size, video=None):
        if video is not None:
            seconds = video_seconds(video)
        elif frames > 0:
            seconds = frames / fps
        audio = silent_audio(seconds, sample_rate, channels, batch_size)
        return (audio, float(seconds), int(audio["waveform"].shape[-1]))


class VideoAddSilentAudio:
    """Attach a silent audio track to a video that has none.

    LanPaint AV Encode decodes the audio out of the VIDEO itself and raises
    "the video has no audio track to encode" on a silent clip. Run the video
    through this node first and AV Encode gets a track exactly as long as the
    picture; pair it with ``AudioMask`` in ``keep_all`` mode so the silence is
    never resampled.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to give a silent audio track."}),
                "sample_rate": ("INT", {
                    "default": 32000, "min": 1000, "max": 192000, "step": 100,
                    "tooltip": "32000 matches the MiniMax H3 audio VAE, so nothing has to be resampled.",
                }),
                "channels": (SilentAudio.CHANNELS, {"default": "stereo"}),
                "replace_existing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: a video that already has audio is passed through untouched.",
                }),
            },
        }

    RETURN_TYPES = ("VIDEO", "AUDIO")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "run"
    CATEGORY = f"{CATEGORY}/av"
    DESCRIPTION = ("Give a soundless video a silent audio track of the same length, "
                   "so LanPaint AV Encode can encode it.")

    def run(self, video, sample_rate, channels, replace_existing):
        if VideoFromComponents is None or VideoComponents is None:
            raise RuntimeError("this node requires the ComfyUI runtime (comfy_api)")

        components = video.get_components()
        existing = getattr(components, "audio", None)
        if existing is not None and not replace_existing:
            return (video, existing)

        rate = components.frame_rate
        if not isinstance(rate, Fraction):
            rate = Fraction(float(rate)).limit_denominator(1000000)
        frames = int(components.images.shape[0])
        audio = silent_audio(frames / float(rate), sample_rate, channels)

        patched = VideoComponents(
            images=components.images,
            frame_rate=rate,
            audio=audio,
            metadata=getattr(components, "metadata", None),
        )
        return (VideoFromComponents(patched), audio)


NODE_CLASS_MAPPINGS = {
    "ToolsAudioMask": AudioMask,
    "ToolsSilentAudio": SilentAudio,
    "ToolsVideoAddSilentAudio": VideoAddSilentAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsAudioMask": "Audio Mask (tools)",
    "ToolsSilentAudio": "Silent Audio (tools)",
    "ToolsVideoAddSilentAudio": "Video Add Silent Audio (tools)",
}
