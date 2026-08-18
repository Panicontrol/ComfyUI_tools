from fractions import Fraction

import pytest
import torch

from comfyui_tools import av_nodes
from comfyui_tools.av_nodes import (
    AudioMask,
    SilentAudio,
    VideoAddSilentAudio,
    parse_intervals,
    silent_audio,
    video_frame_info,
)


class FakeVideo:
    """Minimal stand-in for a ComfyUI VIDEO input."""

    def __init__(self, frames=48, fps=24.0):
        self._frames = frames
        self._fps = fps

    def get_frame_count(self):
        return self._frames

    def get_fps(self):
        return self._fps


class ComponentsOnlyVideo:
    """A VIDEO that only exposes get_components(), like VideoFromComponents."""

    class _Components:
        images = torch.zeros((12, 4, 4, 3))
        frame_rate = 12.0

    def get_components(self):
        return self._Components()


def build(**kwargs):
    params = {
        "mode": "keep_all",
        "intervals": "",
        "frames": 81,
        "fps": 25.0,
        "invert": False,
        "output_shape": "[F]",
    }
    params.update(kwargs)
    return AudioMask().build(**params)


def test_keep_all_is_an_all_zero_stub():
    mask, frames, fps = build(frames=16)
    assert mask.shape == (16,)
    assert mask.dtype == torch.float32
    assert mask.max() == 0.0
    assert (frames, fps) == (16, 25.0)


def test_regenerate_all():
    mask, _, _ = build(mode="regenerate_all", frames=8)
    assert mask.min() == 1.0


def test_invert_turns_the_stub_into_a_full_mask():
    mask, _, _ = build(frames=8, invert=True)
    assert mask.min() == 1.0


def test_output_shape_f1():
    mask, _, _ = build(frames=8, output_shape="[F, 1]")
    assert mask.shape == (8, 1)


def test_frame_count_comes_from_the_video_mask():
    video_mask = torch.zeros((33, 64, 64))
    mask, frames, _ = build(frames=81, video_mask=video_mask)
    assert mask.shape == (33,) and frames == 33


def test_video_supplies_frames_and_fps():
    mask, frames, fps = build(video=FakeVideo(48, 24.0))
    assert mask.shape == (48,)
    assert (frames, fps) == (48, 24.0)


def test_video_mask_wins_over_the_video():
    _, frames, fps = build(video=FakeVideo(48, 24.0), video_mask=torch.zeros((10, 8, 8)))
    assert frames == 10
    assert fps == 24.0


def test_video_frame_info_falls_back_to_components():
    assert video_frame_info(ComponentsOnlyVideo()) == (12, 12.0)


def test_intervals_mark_only_the_listed_seconds():
    mask, _, _ = build(mode="intervals", intervals="1.0-2.0", frames=100, fps=10.0)
    assert mask[:10].max() == 0.0
    assert mask[10:20].min() == 1.0
    assert mask[20:].max() == 0.0


def test_intervals_accept_the_editor_json():
    mask, _, _ = build(
        mode="intervals",
        intervals='[{"start": 0.0, "end": 0.5}, {"start": 1.5, "end": 2.0}]',
        frames=30,
        fps=10.0,
    )
    assert mask[:5].min() == 1.0
    assert mask[5:15].max() == 0.0
    assert mask[15:20].min() == 1.0


def test_intervals_are_clamped_to_the_clip():
    mask, _, _ = build(mode="intervals", intervals="0-99", frames=10, fps=10.0)
    assert mask.min() == 1.0


def test_intervals_past_the_end_of_the_clip_change_nothing():
    mask, _, _ = build(mode="intervals", intervals="20-30", frames=10, fps=10.0)
    assert mask.max() == 0.0


def test_negative_json_start_is_clamped_to_zero():
    mask, _, _ = build(
        mode="intervals",
        intervals='[{"start": -2.0, "end": 0.5}]',
        frames=10,
        fps=10.0,
    )
    assert mask[:5].min() == 1.0
    assert mask[5:].max() == 0.0


def test_empty_and_reversed_intervals_are_ignored():
    mask, _, _ = build(mode="intervals", intervals="2.0-1.0", frames=10, fps=10.0)
    assert mask.max() == 0.0


def test_intervals_ignore_a_bare_mode_switch():
    mask, _, _ = build(mode="intervals", intervals="", frames=10, fps=10.0)
    assert mask.max() == 0.0


def test_parse_intervals_supports_several_separators():
    assert parse_intervals("0-1, 2:3\n4 5;6-7") == [
        (0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)
    ]


def test_parse_intervals_rejects_garbage():
    with pytest.raises(ValueError):
        parse_intervals("nonsense")
    with pytest.raises(ValueError):
        parse_intervals("[{broken}]")


def test_zero_frames_is_rejected():
    with pytest.raises(ValueError):
        build(frames=0)


class FakeComponents:
    def __init__(self, images, frame_rate, audio=None, metadata=None):
        self.images = images
        self.frame_rate = frame_rate
        self.audio = audio
        self.metadata = metadata


class FakeVideoWithComponents:
    """A VIDEO whose components can carry (or lack) an audio track."""

    def __init__(self, frames=10, frame_rate=Fraction(10, 1), audio=None):
        self.components = FakeComponents(
            torch.zeros((frames, 8, 8, 3)), frame_rate, audio
        )

    def get_components(self):
        return self.components


@pytest.fixture
def comfy_api(monkeypatch):
    """Stand in for comfy_api's VideoComponents / VideoFromComponents."""
    built = {}

    def from_components(components):
        built["components"] = components
        return components

    monkeypatch.setattr(av_nodes, "VideoComponents", FakeComponents)
    monkeypatch.setattr(av_nodes, "VideoFromComponents", from_components)
    return built


def test_silent_audio_helper_shape_and_rate():
    audio = silent_audio(2.0, 32000, "stereo")
    assert audio["sample_rate"] == 32000
    assert audio["waveform"].shape == (1, 2, 64000)
    assert audio["waveform"].abs().max() == 0.0


def test_silent_audio_helper_mono_and_batch():
    audio = silent_audio(0.5, 16000, "mono", batch_size=3)
    assert audio["waveform"].shape == (3, 1, 8000)


def test_silent_audio_helper_rounds_up_to_a_whole_sample():
    assert silent_audio(0.00001, 32000, "mono")["waveform"].shape[-1] == 1


def test_silent_audio_helper_rejects_empty_audio():
    with pytest.raises(ValueError):
        silent_audio(0.0, 32000, "stereo")
    with pytest.raises(ValueError):
        silent_audio(1.0, 0, "stereo")


def test_silent_audio_node_uses_seconds():
    audio, seconds, samples = SilentAudio().generate(2.0, 0, 25.0, 32000, "stereo", 1)
    assert (seconds, samples) == (2.0, 64000)
    assert audio["waveform"].shape == (1, 2, 64000)


def test_silent_audio_node_uses_the_frame_count():
    _, seconds, samples = SilentAudio().generate(2.0, 50, 25.0, 32000, "mono", 1)
    assert seconds == 2.0 and samples == 64000


def test_silent_audio_node_takes_the_duration_from_the_video():
    _, seconds, _ = SilentAudio().generate(2.0, 0, 25.0, 32000, "stereo", 1,
                                           video=FakeVideo(48, 24.0))
    assert seconds == 2.0


def test_add_silent_audio_matches_the_video_length(comfy_api):
    video = FakeVideoWithComponents(frames=30, frame_rate=Fraction(30, 1))
    patched, audio = VideoAddSilentAudio().run(video, 32000, "stereo", False)

    assert audio["waveform"].shape == (1, 2, 32000)  # 30 frames @ 30 fps = 1 s
    assert patched is comfy_api["components"]
    assert patched.audio is audio
    assert patched.frame_rate == Fraction(30, 1)
    assert patched.images is video.components.images


def test_add_silent_audio_accepts_a_float_frame_rate(comfy_api):
    video = FakeVideoWithComponents(frames=12, frame_rate=24.0)
    _, audio = VideoAddSilentAudio().run(video, 16000, "mono", False)
    assert audio["waveform"].shape == (1, 1, 8000)  # 0.5 s


def test_add_silent_audio_keeps_an_existing_track(comfy_api):
    existing = silent_audio(1.0, 44100, "stereo")
    video = FakeVideoWithComponents(audio=existing)
    result, audio = VideoAddSilentAudio().run(video, 32000, "stereo", False)
    assert result is video and audio is existing
    assert "components" not in comfy_api


def test_add_silent_audio_can_replace_an_existing_track(comfy_api):
    existing = silent_audio(1.0, 44100, "stereo")
    video = FakeVideoWithComponents(frames=10, frame_rate=Fraction(10, 1), audio=existing)
    _, audio = VideoAddSilentAudio().run(video, 32000, "stereo", True)
    assert audio is not existing
    assert audio["sample_rate"] == 32000


def test_add_silent_audio_needs_the_comfy_runtime(monkeypatch):
    monkeypatch.setattr(av_nodes, "VideoFromComponents", None)
    with pytest.raises(RuntimeError):
        VideoAddSilentAudio().run(FakeVideoWithComponents(), 32000, "stereo", False)
