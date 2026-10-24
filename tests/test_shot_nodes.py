import json
import os

import pytest
import torch

from comfyui_tools.image_nodes import SaveImageSequence
from comfyui_tools.shot_nodes import ShotFinalize, ShotPaths, parse_items


def render(root, stage, name, version, frames=2, latent=False, prompt=None):
    """What Save Image Sequence writes for one stage version."""
    folder = os.path.join(root, "shot_01", "_tmp", stage)
    kwargs = {"latent": {"samples": torch.zeros(1, 4, 2, 2)}} if latent else {}
    SaveImageSequence().save(torch.rand(frames, 8, 8, 3), folder, name, version, 1001, 5, False,
                             prompt=prompt, extra_pnginfo={"workflow": {"id": "wf-1", "nodes": []}}, **kwargs)
    return folder


def paths(root, **kw):
    args = dict(root=str(root), shot="shot_01", stage="02_track", object="fly", source_stage="01_generate",
                source_object="", source_version=0, padding=3)
    args.update(kw)
    return ShotPaths().paths(**args)


def test_shot_paths_layout_latest_version_and_errors(tmp_path):
    # no folders yet: paths are fine, only the source is an error
    save, new, *_ = paths(tmp_path, source_stage="none")
    assert save == os.path.join(str(tmp_path), "shot_01", "_tmp", "02_track") and new == "shot_01_fly"
    with pytest.raises(FileNotFoundError):
        paths(tmp_path)

    render(tmp_path, "01_generate", "shot_01", 1, latent=True)
    render(tmp_path, "01_generate", "shot_01", 2)
    before = ShotPaths.IS_CHANGED(str(tmp_path), "shot_01", "02_track", "fly", "01_generate", "", 0, 3)
    *_, video, latent, frames, version, info = paths(tmp_path)
    assert version == 2 and video.endswith("shot_01_v002.mp4") and frames.endswith("shot_01_v002")
    assert "latest" in info and "no .latent" in info

    render(tmp_path, "01_generate", "shot_01", 3)   # a new run: "latest" must move on
    assert ShotPaths.IS_CHANGED(str(tmp_path), "shot_01", "02_track", "fly", "01_generate", "", 0, 3) != before
    assert paths(tmp_path)[5] == 3

    with pytest.raises(FileNotFoundError, match=r"shot_01_v004.mp4 not found in .*available: v001, v002, v003"):
        paths(tmp_path, source_version=4)
    assert paths(tmp_path, stage="final")[0] == os.path.join(str(tmp_path), "shot_01", "final")


def test_parse_items():
    assert parse_items("04_upscale v003            -> final\n02_track   v004 fly -> mask_fly\n# note\n01_generate 2") == [
        ("04_upscale", 3, "", "final"), ("02_track", 4, "fly", "mask_fly"), ("01_generate", 2, "", "")]
    with pytest.raises(ValueError, match="line 1"):
        parse_items("05_grade v001")


def finalize(root, **kw):
    args = dict(root=str(root), shot="shot_01",
                items="04_upscale v003 -> final\n02_track v004 fly -> mask_fly\n01_generate v002 -> base",
                final_version=0, copy_frames=True, copy_latents=False, write_manifest=True,
                delete_tmp=False, confirm_delete="", dry_run=True)
    args.update(kw)
    return ShotFinalize().finalize(**args)["result"]


def test_shot_finalize_dry_run_copy_versions_and_delete(tmp_path):
    prompt = {"3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 8, "cfg": 1.0, "denoise": 0.6}},
              "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fly lands on the cream"}}}
    render(tmp_path, "04_upscale", "shot_01", 3, prompt=prompt)
    render(tmp_path, "02_track", "shot_01_fly", 4)
    render(tmp_path, "01_generate", "shot_01", 2)
    shot = tmp_path / "shot_01"

    dest, report = finalize(tmp_path)                       # dry run: nothing on disk
    assert "PLAN" in report and dest.endswith("shot_01_final_v001") and not (shot / "final").exists()

    dest, report = finalize(tmp_path, dry_run=False, delete_tmp=True, confirm_delete="wrong")
    names = sorted(os.listdir(dest))
    assert names == ["base", "base.mp4", "final", "final.mp4", "manifest.md", "mask_fly", "mask_fly.mp4"]
    manifest = (shot / "final" / "shot_01_final_v001" / "manifest.md").read_text(encoding="utf-8")
    assert "seed=42" in manifest and "denoise=0.6" in manifest and "a fly lands on the cream" in manifest
    assert (shot / "_tmp").is_dir() and "confirm_delete is not the shot name" in report

    marker = shot / "final" / "shot_01_final_v001" / "base.mp4"
    stamp = marker.stat().st_mtime
    dest2, report = finalize(tmp_path, dry_run=False, copy_frames=False, delete_tmp=True, confirm_delete="shot_01")
    assert dest2.endswith("shot_01_final_v002") and marker.stat().st_mtime == stamp   # v001 untouched
    assert not (shot / "_tmp").exists() and (shot / "final" / "shot_01_final_v002" / "final.mp4").exists()

    with pytest.raises(FileExistsError):
        finalize(tmp_path, dry_run=False, final_version=1)


def test_shot_finalize_checks_everything_before_copying(tmp_path):
    render(tmp_path, "04_upscale", "shot_01", 3)
    with pytest.raises(FileNotFoundError, match=r"02_track v004 fly: shot_01_fly_v004.mp4 not found"):
        finalize(tmp_path, dry_run=False)
    assert not (tmp_path / "shot_01" / "final").exists()


def test_merge_and_upscale_stages(tmp_path):
    from comfyui_tools.shot_nodes import STAGES

    assert STAGES.index("04_merge") < STAGES.index("05_upscale") < STAGES.index("04_upscale")
    for stage in ("04_merge", "05_upscale"):
        save = paths(tmp_path, stage=stage, object="", source_stage="none")[0]
        assert save == os.path.join(str(tmp_path), "shot_01", "_tmp", stage)

    render(tmp_path, "04_merge", "shot_01", 1, latent=True)
    render(tmp_path, "04_merge", "shot_01", 2, latent=True)
    *_, video, latent, frames, version, info = paths(tmp_path, stage="05_upscale", object="", source_stage="04_merge")
    assert version == 2 and latent.endswith(os.path.join("04_merge", "shot_01_v002.latent")) and os.path.isfile(latent)

    render(tmp_path, "05_upscale", "shot_01", 1, latent=True)
    assert parse_items("05_upscale v001 -> final\n04_merge v002 -> merged") == [
        ("05_upscale", 1, "", "final"), ("04_merge", 2, "", "merged")]
    dest, report = finalize(tmp_path, items="05_upscale v001 -> final\n04_merge v002 -> merged",
                            copy_latents=True, dry_run=False)
    assert sorted(os.listdir(dest)) == ["final", "final.latent", "final.mp4", "manifest.md", "merged", "merged.latent", "merged.mp4"]
