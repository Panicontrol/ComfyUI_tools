import json
import os

import pytest
import torch

from comfyui_tools.image_nodes import SaveImageSequence
from comfyui_tools.shot_nodes import ShotFinalize, ShotPaths, parse_items


def render(root, stage, name, version, frames=2, latent=False, prompt=None, notes=None):
    """What Save Image Sequence writes for one stage version."""
    folder = os.path.join(root, "shot_01", "_tmp", stage)
    kwargs = {"latent": {"samples": torch.zeros(1, 4, 2, 2)}} if latent else {}
    if notes is not None:
        kwargs["notes"] = notes
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
    save, new, *_, info = paths(tmp_path, source_stage="none")
    assert save == os.path.join(str(tmp_path), "shot_01", "_tmp", "02_track") and new == "shot_01_fly"
    assert info == "02_track <- none"
    with pytest.raises(FileNotFoundError, match="available: none"):
        paths(tmp_path)

    render(tmp_path, "01_generate", "shot_01", 1, latent=True)
    render(tmp_path, "01_generate", "shot_01", 2)
    before = ShotPaths.IS_CHANGED(str(tmp_path), "shot_01", "02_track", "fly", "01_generate", "", 0, 3)
    *_, video, latent, frames, version, info = paths(tmp_path)
    assert version == 2 and video.endswith("shot_01_v002.mp4") and frames.endswith("shot_01_v002")
    assert info == "02_track <- 01_generate shot_01_v002 (no latent)"
    assert paths(tmp_path, source_version=1)[6] == "02_track <- 01_generate shot_01_v001"

    render(tmp_path, "01_generate", "shot_01", 3)   # a new run: "latest" must move on
    assert ShotPaths.IS_CHANGED(str(tmp_path), "shot_01", "02_track", "fly", "01_generate", "", 0, 3) != before
    assert paths(tmp_path)[5] == 3

    with pytest.raises(FileNotFoundError, match=r"shot_01_v004.latent / .mp4 not found in .*available: v001, v002, v003"):
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
    with pytest.raises(FileNotFoundError, match=r"02_track v004 fly: shot_01_fly_v004.latent / .mp4 not found"):
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


def test_shot_paths_base_chain(tmp_path):
    base = os.path.join(str(tmp_path), "shot_01", "_tmp", "base")
    save, new, *_, info = paths(tmp_path, stage="base", object="", source_stage="none")
    assert save == base and new == "shot_01" and info == "base <- none"
    # a detail pass reads the latest base and saves the next one into the same folder
    with pytest.raises(FileNotFoundError, match=r"base <- base: no shot_01_v###.latent / .mp4 .*available: none"):
        paths(tmp_path, stage="base", object="", source_stage="base")

    render(tmp_path, "base", "shot_01", 1, latent=True)
    render(tmp_path, "base", "shot_01", 2, latent=True)
    save, new, video, latent, _, version, info = paths(tmp_path, stage="base", object="", source_stage="base")
    assert (save, new, version) == (base, "shot_01", 2) and latent == os.path.join(base, "shot_01_v002.latent")
    assert info == "base <- base shot_01_v002"
    assert paths(tmp_path, stage="base", object="", source_stage="base", source_version=1)[3].endswith("shot_01_v001.latent")

    # base has no objects: given ones are ignored, and info says so
    _, new, *_, info = paths(tmp_path, stage="base", object="fly", source_stage="base", source_object="fly")
    assert new == "shot_01" and info.startswith("base <- base shot_01_v002 (")
    assert "object 'fly' ignored" in info and "source_object 'fly' ignored" in info

    render(tmp_path, "03_detail", "shot_01_fly", 1)
    *_, video, _, _, _, info = paths(tmp_path, stage="base", object="", source_stage="03_detail", source_object="fly")
    assert video.endswith(os.path.join("03_detail", "shot_01_fly_v001.mp4"))
    assert info == "base <- 03_detail shot_01_fly_v001 (no latent)"
    assert parse_items("base v003 -> final\nbase v001 -> generation") == [("base", 3, "", "final"), ("base", 1, "", "generation")]


def mp4_tags(path):
    import av

    with av.open(path) as container:
        return dict(container.metadata)


def test_save_notes(tmp_path):
    folder = render(tmp_path, "base", "shot_01", 1, notes="")
    assert not os.path.exists(os.path.join(folder, "shot_01_v001.notes.txt"))
    tags = mp4_tags(os.path.join(folder, "shot_01_v001.mp4"))
    assert "notes" not in tags and json.loads(tags["workflow"])["id"] == "wf-1"
    render(tmp_path, "base", "shot_01", 2, notes="  \n")   # whitespace only is empty too
    assert not os.path.exists(os.path.join(folder, "shot_01_v002.notes.txt"))

    notes = "base <- base shot_01_v002\nbase <- 03_detail shot_01_fly_v001\nbase <- 02_track shot_01_fly_v001"
    render(tmp_path, "base", "shot_01", 3, notes=notes)
    tags = mp4_tags(os.path.join(folder, "shot_01_v003.mp4"))
    assert tags["notes"] == "shot_01_v003\n" + notes and "workflow" in tags
    text = open(os.path.join(folder, "shot_01_v003.notes.txt"), encoding="utf-8").read()
    assert text.splitlines()[0] == "shot_01_v003" and text.strip() == tags["notes"]

    # an existing notes file is not overwritten either
    os.remove(os.path.join(folder, "shot_01_v003.mp4"))
    with pytest.raises(ValueError, match="shot_01_v003.notes.txt already exists"):
        render(tmp_path, "base", "shot_01", 3, notes=notes)


def test_shot_finalize_base_history(tmp_path):
    render(tmp_path, "base", "shot_01", 1, notes="base <- none")
    render(tmp_path, "base", "shot_01", 2)                     # no notes in the mp4 ...
    folder = render(tmp_path, "base", "shot_01", 3, notes="base <- base shot_01_v002\nbase <- 03_detail shot_01_face_v001")
    with open(os.path.join(folder, "shot_01_v002.notes.txt"), "w", encoding="utf-8") as handle:
        handle.write("shot_01_v002\nbase <- 03_detail shot_01_fly_v001\n")   # ... but a notes file
    render(tmp_path, "base", "shot_01", 5, notes="base <- base shot_01_v003")   # v004 is missing

    dest, report = finalize(tmp_path, items="base v005 -> final\nbase v001 -> generation", dry_run=False, copy_frames=False)
    assert sorted(os.listdir(dest)) == ["final.mp4", "final.notes.txt", "generation.mp4", "generation.notes.txt", "manifest.md"]
    manifest = open(os.path.join(dest, "manifest.md"), encoding="utf-8").read()
    final, generation = manifest.split("## base v001 -> generation")
    assert "notes:" in final and "base <- base shot_01_v003" in final
    history = final.split("### История базы")[1]
    order = [history.index(s) for s in ("**v001**", "base <- none", "**v002**", "03_detail shot_01_fly_v001",
                                        "**v003**", "03_detail shot_01_face_v001", "**v004**: нет файла", "**v005**")]
    assert order == sorted(order)
    assert "### История базы" in generation and "**v002**" not in generation


def drop_latent(root, stage, name, version):
    """A .latent put into a stage folder by hand, with no mp4 (a base from elsewhere)."""
    from comfyui_tools.latent_nodes import write_latent

    folder = os.path.join(str(root), "shot_01", "_tmp", stage)
    os.makedirs(folder, exist_ok=True)
    write_latent(os.path.join(folder, f"{name}_v{version:03d}.latent"), {"samples": torch.zeros(1, 4, 2, 2)})
    return folder


def test_shot_paths_finds_versions_by_latent_or_mp4(tmp_path):
    folder = drop_latent(tmp_path, "base", "shot_01", 1)
    save, new, video, latent, frames, version, info = paths(tmp_path, stage="base", object="", source_stage="base")
    assert version == 1 and info == "base <- base shot_01_v001 (no mp4)"
    assert video == os.path.join(folder, "shot_01_v001.mp4") and not os.path.exists(video)  # the path all the same
    assert latent == os.path.join(folder, "shot_01_v001.latent") and frames == os.path.join(folder, "shot_01_v001")

    render(tmp_path, "base", "shot_01", 2, latent=True)
    before = ShotPaths.IS_CHANGED(str(tmp_path), "shot_01", "base", "", "base", "", 0, 3)
    drop_latent(tmp_path, "base", "shot_01", 3)                      # latest = a .latent alone
    assert ShotPaths.IS_CHANGED(str(tmp_path), "shot_01", "base", "", "base", "", 0, 3) != before
    assert paths(tmp_path, stage="base", object="", source_stage="base")[5:] == (3, "base <- base shot_01_v003 (no mp4)")
    assert paths(tmp_path, stage="base", object="", source_stage="base", source_version=2)[6] == "base <- base shot_01_v002"
    with pytest.raises(FileNotFoundError, match=r"shot_01_v004.latent / .mp4 not found .*available: v001, v002, v003"):
        paths(tmp_path, stage="base", object="", source_stage="base", source_version=4)


def test_shot_finalize_item_without_mp4(tmp_path):
    folder = drop_latent(tmp_path, "base", "shot_01", 1)            # no notes at all
    drop_latent(tmp_path, "base", "shot_01", 2)
    with open(os.path.join(folder, "shot_01_v002.notes.txt"), "w", encoding="utf-8") as handle:
        handle.write("shot_01_v002\nbase <- base shot_01_v001\n")
    render(tmp_path, "base", "shot_01", 3, latent=True, notes="base <- base shot_01_v002")

    # copy_latents off and copy_frames on: the .latent is still copied, and a missing PNG folder is fine
    dest, report = finalize(tmp_path, items="base v002 -> final\nbase v003 -> preview", dry_run=False)
    assert sorted(os.listdir(dest)) == ["final.latent", "final.notes.txt", "manifest.md",
                                        "preview", "preview.mp4", "preview.notes.txt"]
    manifest = open(os.path.join(dest, "manifest.md"), encoding="utf-8").read()
    final, preview = manifest.split("## base v003 -> preview.mp4")
    assert "## base v002 -> final.latent" in final and "- mp4 нет" in final and "- mp4 нет" not in preview
    assert "**v001**: нет notes" in final and "base <- base shot_01_v001" in final
    assert "**v002**: нет" not in preview and "base <- base shot_01_v002" in preview
