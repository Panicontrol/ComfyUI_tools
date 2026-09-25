import numpy as np
import pytest
import torch
from PIL import Image

from comfyui_tools.image_nodes import LoadImageSequence, natural_key


def write_png(folder, name, size=(8, 6), color=(255, 0, 0), alpha=None):
    mode = "RGB" if alpha is None else "RGBA"
    fill = color if alpha is None else (*color, alpha)
    path = folder / name
    Image.new(mode, size, fill).save(path)
    return path


@pytest.fixture
def sequence(tmp_path):
    for index in (1, 2, 10):  # out of order on a plain string sort
        write_png(tmp_path, f"frame_{index}.png", color=(index, index, index))
    return tmp_path


def load(directory, **kwargs):
    params = {
        "directory": str(directory),
        "pattern": "*.png",
        "sort_by": "name",
        "reverse": False,
        "start_index": 0,
        "step": 1,
        "count": 0,
        "on_size_mismatch": "error",
    }
    params.update(kwargs)
    return LoadImageSequence().load(**params)


def test_natural_key_orders_frames_like_a_sequence():
    names = sorted(["frame_10.png", "frame_2.png", "frame_1.png"], key=natural_key)
    assert names == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_loads_a_folder_as_one_batch(sequence):
    images, masks, count, filenames = load(sequence)
    assert images.shape == (3, 6, 8, 3)
    assert images.dtype == torch.float32
    assert masks.shape == (3, 6, 8)
    assert count == 3
    assert filenames.splitlines() == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_pixels_are_normalised(tmp_path):
    write_png(tmp_path, "a.png", color=(255, 128, 0))
    images, _, _, _ = load(tmp_path)
    assert images.max() <= 1.0
    assert images[0, 0, 0, 0] == pytest.approx(1.0)
    assert images[0, 0, 0, 1] == pytest.approx(128 / 255, abs=1e-4)


def test_alpha_becomes_the_mask(tmp_path):
    write_png(tmp_path, "opaque.png", alpha=255)
    write_png(tmp_path, "clear.png", alpha=0)
    _, masks, _, names = load(tmp_path)
    order = names.splitlines()
    assert masks[order.index("opaque.png")].max() == 0.0  # 1 - alpha
    assert masks[order.index("clear.png")].min() == 1.0


def test_a_frame_without_alpha_gets_an_empty_mask(sequence):
    _, masks, _, _ = load(sequence)
    assert masks.max() == 0.0


def test_start_step_and_count(sequence):
    assert load(sequence, start_index=1)[3].splitlines() == ["frame_2.png", "frame_10.png"]
    assert load(sequence, step=2)[3].splitlines() == ["frame_1.png", "frame_10.png"]
    assert load(sequence, count=2)[3].splitlines() == ["frame_1.png", "frame_2.png"]
    assert load(sequence, start_index=1, count=1)[3].splitlines() == ["frame_2.png"]


def test_reverse(sequence):
    assert load(sequence, reverse=True)[3].splitlines()[0] == "frame_10.png"


def test_sort_by_modified(tmp_path):
    import os

    first = write_png(tmp_path, "b.png")
    second = write_png(tmp_path, "a.png")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    assert load(tmp_path, sort_by="modified")[3].splitlines() == ["b.png", "a.png"]


def test_pattern_filters_and_accepts_several_globs(tmp_path):
    write_png(tmp_path, "frame_1.png")
    write_png(tmp_path, "render_1.png")
    write_png(tmp_path, "note.jpg")
    assert load(tmp_path, pattern="frame_*.png")[2] == 1
    assert load(tmp_path, pattern="frame_*.png, render_*.png")[2] == 2
    assert load(tmp_path, pattern="*")[2] == 3


def test_pattern_is_case_insensitive(tmp_path):
    write_png(tmp_path, "FRAME_1.PNG")
    assert load(tmp_path, pattern="*.png")[2] == 1


def test_subfolders_are_left_out(tmp_path):
    write_png(tmp_path, "frame_1.png")
    (tmp_path / "nested").mkdir()
    write_png(tmp_path / "nested", "frame_2.png")
    assert load(tmp_path)[2] == 1


def test_grayscale_is_converted_to_rgb(tmp_path):
    Image.new("L", (4, 4), 128).save(tmp_path / "gray.png")
    images, _, _, _ = load(tmp_path)
    assert images.shape == (1, 4, 4, 3)
    assert images[0, 0, 0, 0] == pytest.approx(images[0, 0, 0, 2])


def test_size_mismatch_is_an_error_by_default(tmp_path):
    write_png(tmp_path, "a.png", size=(8, 6))
    write_png(tmp_path, "b.png", size=(4, 4))
    with pytest.raises(ValueError, match="on_size_mismatch"):
        load(tmp_path)


def test_size_mismatch_can_resize(tmp_path):
    write_png(tmp_path, "a.png", size=(8, 6))
    write_png(tmp_path, "b.png", size=(4, 4), alpha=0)
    images, masks, count, _ = load(tmp_path, on_size_mismatch="resize")
    assert images.shape == (2, 6, 8, 3)
    assert masks.shape == (2, 6, 8)
    assert count == 2


def test_size_mismatch_can_skip(tmp_path):
    write_png(tmp_path, "a.png", size=(8, 6))
    write_png(tmp_path, "b.png", size=(4, 4))
    images, _, count, names = load(tmp_path, on_size_mismatch="skip")
    assert images.shape == (1, 6, 8, 3)
    assert count == 1 and names == "a.png"


def test_missing_folder_and_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="not a folder"):
        load(tmp_path / "nope")
    with pytest.raises(ValueError, match="no files matching"):
        load(tmp_path)
    with pytest.raises(ValueError, match="folder to load"):
        load("")


def test_slicing_past_the_end_explains_itself(sequence):
    with pytest.raises(ValueError, match="skip past every file"):
        load(sequence, start_index=99)


def test_is_changed_follows_the_folder_contents(sequence):
    args = (str(sequence), "*.png", "name", False, 0, 1, 0, "error")
    before = LoadImageSequence.IS_CHANGED(*args)
    assert LoadImageSequence.IS_CHANGED(*args) == before
    write_png(sequence, "frame_11.png")
    assert LoadImageSequence.IS_CHANGED(*args) != before


def test_is_changed_on_a_missing_folder(tmp_path):
    value = LoadImageSequence.IS_CHANGED(str(tmp_path / "nope"), "*.png", "name", False, 0, 1, 0, "error")
    assert value.startswith("missing:")


def test_quotes_and_user_paths_are_accepted(sequence):
    assert load(f'"{sequence}"')[2] == 3
