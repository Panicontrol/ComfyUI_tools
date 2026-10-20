import av
import numpy as np
import pytest

from comfyui_tools.video_nodes import VideoReel, parse_reel_items


def clip(path, frames, width, height, value):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=24)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for _ in range(frames):
            frame = np.full((height, width, 3), value, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def build(items, output, **kwargs):
    params = dict(width=64, height=36, frame_step=1, frame_rate=24.0, font_size=8,
                  box_opacity=0.55, crf=18, draw_stamp=True, overwrite=False)
    params.update(kwargs)
    return VideoReel().build(items, str(output), **params)["result"]


def test_parse_items():
    text = '"C:\\a.mp4"\r\ntest v000   prompt PU008\r\nLoRA: x\r\n\r\n\r\nC:\\b.mp4\n'
    assert parse_reel_items(text) == [("C:\\a.mp4", ["test v000   prompt PU008", "LoRA: x"]),
                                      ("C:\\b.mp4", [])]


def test_reel_concat_step_and_letterbox(tmp_path):
    a = clip(tmp_path / "a.mp4", 6, 64, 36, 200)
    b = clip(tmp_path / "b.mp4", 5, 32, 32, 100)  # other aspect: fitted on black
    out = tmp_path / "reel" / "reel_12fps.mp4"
    video, frames = build(f"{a}\nstamp line\n\n{b}\nother\nlines", out, frame_step=2, frame_rate=12.0)
    assert (video, frames) == (str(out), 3 + 3)
    with av.open(video) as container:
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    assert len(decoded) == 6 and decoded[0].shape == (36, 64, 3)
    assert decoded[-1][18, 2].max() < 20  # side bar of the square clip is black

    with pytest.raises(ValueError, match="already exists"):
        build(f"{a}\n", out)


def test_missing_clips_listed_together(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        build(f"{tmp_path / 'x.mp4'}\n\n{tmp_path / 'y.mp4'}\n", tmp_path / "o.mp4")
    assert "x.mp4" in str(error.value) and "y.mp4" in str(error.value)
