"""Shot folder nodes for a staged pipeline (generate -> track -> detail -> merge -> upscale -> final).

Layout, shared by both nodes:

    <root>/<shot>/_tmp/<stage>/<name>_vNNN.mp4 | .latent | .json | .notes.txt | <name>_vNNN/ (PNG)
    <root>/<shot>/final/<shot>_final_vNNN/      picked results + manifest.md

<name> is <shot> or <shot>_<object>, the way Save Image Sequence names its
files (<new_folder>_vNNN). base is the shot's one versioned chain, always
<shot>_vNNN: generation writes v001 and every detail pass reads the latest
version and writes the next. Shot Paths hands every stage its folders and
finds its source; Shot Finalize copies picked results into a new final folder
and, only when confirmed, deletes _tmp.
"""

import datetime
import json
import os
import re
import shutil

from .utils import CATEGORY

# 01_generate, 04_merge and 04_upscale stay so graphs saved with them still load
STAGES = ["base", "01_generate", "02_track", "03_detail", "04_merge", "05_upscale", "04_upscale"]
DEFAULT_ROOT = r"G:\.shortcut-targets-by-id\1pqdUrxPtcM0AFypHllKSnsTj8jXXqOtJ\26_project\mini_latent_control"


def clean_path(text):
    return os.path.expanduser(str(text).strip().strip('"').rstrip("\\/")) or ""


def stem_name(shot, obj=""):
    shot, obj = str(shot).strip(), str(obj).strip()
    if not shot:
        raise ValueError("give the node a shot name")
    return f"{shot}_{obj}" if obj else shot


def stage_name(shot, stage, obj=""):
    """<name> of a stage's files; base has no objects, so obj is dropped there."""
    return stem_name(shot, "" if stage == "base" else obj)


def ignored_object(stage, obj, field):
    """Note for info when an object is given to base, which has none."""
    obj = str(obj).strip()
    return f"{field} '{obj}' ignored: base has no objects" if stage == "base" and obj else ""


def stage_folder(root, shot, stage):
    return os.path.join(clean_path(root), str(shot).strip(), "_tmp", stage)


def versions_of(folder, name, ext=".mp4"):
    """{version: path} of <name>_vNNN<ext> files in folder (any number of digits)."""
    pattern = re.compile(re.escape(name) + r"_v(\d+)" + re.escape(ext) + r"$", re.IGNORECASE)
    found = {}
    if os.path.isdir(folder):
        for entry in os.listdir(folder):
            m = pattern.match(entry)
            if m:
                found[int(m.group(1))] = os.path.join(folder, entry)
    return found


def version_list(found, padding=3):
    return ", ".join(f"v{v:0{padding}d}" for v in sorted(found)) or "none"


class ShotPaths:
    """Every path one stage of one shot needs, so a graph sets the shot in one place.

    save_folder / new_folder go to Save Image Sequence; source_* point at the
    output of source_stage (0 = its latest version). stage base with source
    base is a detail pass: it reads the latest base and Save writes the next
    one. info is one line, '<stage> <- <source_stage> <file name>', for the
    notes of Save Image Sequence. Creates no folders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root": ("STRING", {"default": DEFAULT_ROOT, "tooltip": "Projects root."}),
                "shot": ("STRING", {"default": "shot_01"}),
                "stage": (STAGES + ["final"], {"default": "01_generate",
                                               "tooltip": "Where this graph writes."}),
                "object": ("STRING", {"default": "",
                                      "tooltip": "Object name (stages 02/03); empty = none. Ignored for base."}),
                "source_stage": (["none"] + STAGES, {"default": "none",
                                                     "tooltip": "Stage this graph reads its input from."}),
                "source_object": ("STRING", {"default": "",
                                             "tooltip": "Object of the source, if it is per object (02_track, "
                                                        "03_detail). Ignored for base."}),
                "source_version": ("INT", {"default": 0, "min": 0, "max": 9999,
                                           "tooltip": "0 = the latest existing version."}),
                "padding": ("INT", {"default": 3, "min": 1, "max": 6, "tooltip": "Digits in version numbers."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("save_folder", "new_folder", "source_video", "source_latent", "source_frames",
                    "source_version", "info")
    FUNCTION = "paths"
    CATEGORY = f"{CATEGORY}/shot"
    DESCRIPTION = "Folders and source files of one pipeline stage of a shot (<root>/<shot>/_tmp/<stage>)."

    @classmethod
    def IS_CHANGED(cls, root, shot, stage, object, source_stage, source_object, source_version, padding):
        if source_stage == "none" or source_version > 0:
            return ""
        # "latest" depends on what is on disk: rerun when the source folder changes
        folder = stage_folder(root, shot, source_stage)
        try:
            return repr(sorted((e.name, e.stat().st_mtime) for e in os.scandir(folder)))
        except OSError:
            return "missing"

    def paths(self, root, shot, stage, object, source_stage, source_object, source_version, padding):
        root, shot = clean_path(root), str(shot).strip()
        if not root:
            raise ValueError("give the node a root folder")
        if stage == "final":
            save_folder = os.path.join(root, shot, "final")
        else:
            save_folder = stage_folder(root, shot, stage)
        new_folder = stage_name(shot, stage, object)
        notes = [ignored_object(stage, object, "object")]

        video = latent = frames = ""
        version = 0
        info = f"{stage} <- none"
        if source_stage != "none":
            folder = stage_folder(root, shot, source_stage)
            name = stage_name(shot, source_stage, source_object)
            notes.append(ignored_object(source_stage, source_object, "source_object"))
            found = versions_of(folder, name)
            version = source_version or max(found, default=0)
            video = found.get(version)
            if video is None:
                wanted = f"{name}_v{version:0{padding}d}.mp4 not found" if source_version else f"no {name}_v###.mp4"
                raise FileNotFoundError(f"{stage} <- {source_stage}: {wanted} in {folder}; "
                                        f"available: {version_list(found, padding)}")
            base = os.path.splitext(video)[0]
            latent, frames = base + ".latent", base
            info = f"{stage} <- {source_stage} {os.path.basename(base)}"
        notes = [n for n in notes if n]
        if notes:
            info += f" ({'; '.join(notes)})"
        return (save_folder, new_folder, video, latent, frames, version, info)


ITEM_RE = re.compile(r"^(?P<stage>\S+)\s+v?(?P<version>\d+)(?:\s+(?P<object>[^\s>-][^\s]*))?\s*(?:->\s*(?P<target>\S+))?\s*$")


def parse_items(text):
    """'base v003 -> final', '02_track v004 fly -> mask_fly' -> [(stage, version, object, target)]."""
    items = []
    for number, raw in enumerate(str(text).replace("\r\n", "\n").split("\n"), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = ITEM_RE.match(line)
        if not m or m.group("stage") not in STAGES:
            raise ValueError(f"items line {number}: cannot read '{raw.strip()}'; "
                             f"use '<stage> v<NNN> [object] [-> name]', stage one of {', '.join(STAGES)}")
        items.append((m.group("stage"), int(m.group("version")), m.group("object") or "", m.group("target") or ""))
    if not items:
        raise ValueError("items is empty: list the results to take into final, one per line")
    return items


def probe_video(path):
    """(width, height, frames, fps, metadata dict) of an mp4, as far as PyAV can tell."""
    try:
        import av

        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            frames = stream.frames or 0
            if not frames and stream.duration and stream.time_base:
                frames = round(float(stream.duration * stream.time_base) * fps)
            return stream.width, stream.height, frames, fps, dict(container.metadata)
    except Exception:
        return None


def read_notes(video, meta=None):
    """Notes Save Image Sequence stored with a video: its mp4 'notes' tag, else <name>.notes.txt; '' if none."""
    text = (meta or {}).get("notes", "")
    path = os.path.splitext(video)[0] + ".notes.txt"
    if not text.strip() and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    return text.strip()


def notes_block(text):
    return ["```text", text, "```"]


def prompt_summary(meta):
    """Lines describing a ComfyUI API prompt stored in mp4 metadata: models, sampling, prompt texts."""
    lines = []
    workflow = meta.get("workflow")
    if workflow:
        try:
            wf = json.loads(workflow)
            lines.append(f"- workflow: id {wf.get('id', '?')}, {len(wf.get('nodes', []))} nodes, "
                         f"format {wf.get('version', '?')}")
        except ValueError:
            pass
    try:
        prompt = json.loads(meta.get("prompt") or "{}")
    except ValueError:
        prompt = {}
    for node_id, node in sorted(prompt.items(), key=lambda kv: str(kv[0])) if isinstance(prompt, dict) else []:
        if not isinstance(node, dict):
            continue
        kind, inputs = node.get("class_type", ""), node.get("inputs", {}) or {}
        plain = {k: v for k, v in inputs.items() if isinstance(v, (str, int, float, bool))}
        if re.search(r"loader|lora", kind, re.IGNORECASE):
            names = {k: v for k, v in plain.items() if isinstance(v, str) and re.search(r"\.(safetensors|ckpt|pt|gguf|pth)$", v)}
            strengths = {k: v for k, v in plain.items() if "strength" in k}
            if names:
                lines.append(f"- {kind} #{node_id}: " + ", ".join(f"{k}={v}" for k, v in {**names, **strengths}.items()))
        elif re.search(r"sampler|sigma|scheduler|noise|guider", kind, re.IGNORECASE):
            keys = ("seed", "noise_seed", "steps", "cfg", "denoise", "sampler_name", "scheduler", "shift")
            picked = {k: plain[k] for k in keys if k in plain}
            if picked:
                lines.append(f"- {kind} #{node_id}: " + ", ".join(f"{k}={v}" for k, v in picked.items()))
        for key, value in plain.items():
            if key in ("text", "prompt", "positive", "negative") and isinstance(value, str) and value.strip():
                text = " ".join(value.split())
                lines.append(f"- {kind} #{node_id} {key}: {text[:600]}{'…' if len(text) > 600 else ''}")
    return lines


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def tree_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files)


class ShotFinalize:
    """Copy picked stage results into <shot>/final/<shot>_final_vNNN and, if confirmed, delete _tmp.

    items: one result per line, '<stage> v<NNN> [object] [-> name]'. The
    manifest carries each result's notes and, for a base version, the notes of
    every base version up to it (what each pass added). Everything
    is checked before anything is copied; an existing final folder is never
    touched; _tmp goes only after every copy worked, only with confirm_delete
    set to the shot name, and only the shot's own _tmp folder (to the recycle
    bin when send2trash is installed). dry_run, on by default, only reports.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root": ("STRING", {"default": DEFAULT_ROOT}),
                "shot": ("STRING", {"default": "shot_01"}),
                "items": ("STRING", {
                    "multiline": True,
                    "default": "base     v003       -> final\nbase     v001       -> generation\n"
                               "02_track v001 fly   -> mask_fly",
                    "tooltip": "One result per line: <stage> v<NNN> [object] [-> name in final]. # starts a comment.",
                }),
                "final_version": ("INT", {"default": 0, "min": 0, "max": 9999,
                                          "tooltip": "0 = next free <shot>_final_vNNN."}),
                "copy_frames": ("BOOLEAN", {"default": True, "tooltip": "Copy the PNG sequences."}),
                "copy_latents": ("BOOLEAN", {"default": False, "tooltip": "Copy the .latent files (large)."}),
                "write_manifest": ("BOOLEAN", {"default": True}),
                "delete_tmp": ("BOOLEAN", {"default": False,
                                           "tooltip": "Delete <shot>/_tmp after a successful copy."}),
                "confirm_delete": ("STRING", {"default": "",
                                              "tooltip": "Deleting only happens when this holds the shot name."}),
                "dry_run": ("BOOLEAN", {"default": True, "tooltip": "Copy and delete nothing, only show the plan."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_folder", "report")
    FUNCTION = "finalize"
    OUTPUT_NODE = True
    CATEGORY = f"{CATEGORY}/shot"
    DESCRIPTION = "Collect picked results of a shot's stages into a new final folder; optionally delete _tmp."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # works on the disk: run every time

    def finalize(self, root, shot, items, final_version=0, copy_frames=True, copy_latents=False,
                 write_manifest=True, delete_tmp=False, confirm_delete="", dry_run=True):
        root, shot = clean_path(root), str(shot).strip()
        if not root or not shot:
            raise ValueError("give the node a root folder and a shot name")
        shot_dir = os.path.join(root, shot)
        final_root = os.path.join(shot_dir, "final")

        # an existing final folder is never touched
        existing = versions_of(final_root, f"{shot}_final", ext="")
        if final_version == 0:
            final_version = max(existing, default=0) + 1
        dest = os.path.join(final_root, f"{shot}_final_v{final_version:03d}")
        if os.path.exists(dest):
            raise FileExistsError(f"{dest} already exists; it is never overwritten -- set final_version to 0")

        # 1. resolve every item before touching the disk
        plan, problems, targets = [], [], set()
        for stage, version, obj, target in parse_items(items):
            folder = stage_folder(root, shot, stage)
            name = stage_name(shot, stage, obj)
            found = versions_of(folder, name)
            ignored = ignored_object(stage, obj, "object")
            label = f"{stage} v{version:03d}" + (f" ({ignored})" if ignored else f" {obj}" if obj else "")
            video = found.get(version)
            if video is None:
                problems.append(f"{label}: {name}_v{version:03d}.mp4 not found in {folder}; "
                                f"available: {version_list(found)}")
                continue
            base = os.path.splitext(video)[0]
            target = target or os.path.basename(base)
            if target in targets:
                problems.append(f"{label}: name '{target}' is used twice in final")
            targets.add(target)
            files = [(video, target + ".mp4")]
            for ext in (".json", ".notes.txt"):
                if os.path.isfile(base + ext):
                    files.append((base + ext, target + ext))
            if copy_latents:
                if os.path.isfile(base + ".latent"):
                    files.append((base + ".latent", target + ".latent"))
                else:
                    problems.append(f"{label}: no {os.path.basename(base)}.latent (turn copy_latents off to skip)")
            if copy_frames:
                if os.path.isdir(base):
                    files.append((base, target))
                else:
                    problems.append(f"{label}: no PNG folder {os.path.basename(base)} (turn copy_frames off to skip)")
            plan.append((label, video, files, version if stage == "base" else 0))
        if problems:
            raise FileNotFoundError("nothing was copied:\n" + "\n".join(problems))

        total = sum(tree_size(src) for _, _, files, _ in plan for src, _ in files)
        tmp_dir = os.path.join(shot_dir, "_tmp")
        wants_delete = delete_tmp and confirm_delete.strip() == shot
        lines = [f"{'PLAN (dry run)' if dry_run else 'DONE'}: {shot} -> {dest}", ""]
        for label, _, files, _ in plan:
            lines.append(label + ":")
            lines += [f"  {os.path.basename(src)} -> {name}{os.sep if os.path.isdir(src) else ''}"
                      f"  ({human_size(tree_size(src))})" for src, name in files]
        lines.append(f"total {human_size(total)}")

        if dry_run:
            lines.append(f"_tmp: {'would be deleted' if wants_delete else 'kept'}"
                         + ("" if wants_delete or not delete_tmp else " (confirm_delete is not the shot name)"))
            report = "\n".join(lines)
            return {"ui": {"text": [report]}, "result": (dest, report)}

        # 2. copy into a new final folder; a failed copy removes the half-made folder and keeps _tmp
        os.makedirs(final_root, exist_ok=True)
        os.makedirs(dest)
        try:
            for _, _, files, _ in plan:
                for src, name in files:
                    out = os.path.join(dest, name)
                    if os.path.isdir(src):
                        shutil.copytree(src, out)
                    else:
                        shutil.copy2(src, out)
            if write_manifest:
                self._manifest(dest, shot, final_version, plan)
        except Exception as error:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(f"copy failed, nothing kept and _tmp untouched: {error}") from error

        # 3. delete _tmp only when confirmed, and only the shot's own _tmp
        if delete_tmp and not wants_delete:
            lines.append("_tmp kept: confirm_delete is not the shot name")
        elif wants_delete:
            real_tmp, real_shot = os.path.realpath(tmp_dir), os.path.realpath(shot_dir)
            if os.path.dirname(real_tmp) != real_shot or os.path.basename(real_tmp) != "_tmp":
                lines.append(f"_tmp kept: {real_tmp} is not {real_shot}{os.sep}_tmp")
            elif not os.path.isdir(real_tmp):
                lines.append("_tmp: nothing to delete")
            else:
                try:
                    from send2trash import send2trash

                    send2trash(real_tmp)
                    lines.append(f"_tmp moved to the recycle bin: {real_tmp}")
                except ImportError:
                    shutil.rmtree(real_tmp)
                    lines.append(f"_tmp deleted (send2trash not installed): {real_tmp}")
                except OSError as error:  # e.g. a virtual drive the recycle bin does not cover
                    shutil.rmtree(real_tmp)
                    lines.append(f"_tmp deleted, the recycle bin refused it ({error}): {real_tmp}")
        else:
            lines.append("_tmp kept")
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (dest, report)}

    @staticmethod
    def _manifest(dest, shot, final_version, plan):
        out = [f"# {shot}_final_v{final_version:03d}", "",
               f"Collected {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
        for label, video, files, base_version in plan:
            out.append(f"## {label} -> {files[0][1]}")
            out.append(f"- source: `{video}`")
            info = probe_video(video)
            meta = info[4] if info else {}
            if info:
                w, h, n, fps, _ = info
                out.append(f"- video: {w}x{h}, {n} frames, {fps:g} fps")
                summary = prompt_summary(meta)
                out += summary if summary else ["- no workflow / prompt in the file's metadata"]
            for src, name in files[1:]:
                out.append(f"- {name}{'/' if os.path.isdir(src) else ''}: {human_size(tree_size(src))}")
            notes = read_notes(video, meta)
            if notes:
                out += ["", "notes:", ""] + notes_block(notes)
            if base_version:
                # what each pass added, generation first
                out += ["", "### История базы", ""]
                found = versions_of(os.path.dirname(video), shot)
                for version in range(1, base_version + 1):
                    path = found.get(version)
                    if path is None:
                        out += [f"**v{version:03d}**: нет файла", ""]
                        continue
                    probed = probe_video(path)
                    text = read_notes(path, probed[4] if probed else {})
                    out += [f"**v{version:03d}**" + ("" if text else ": нет notes"), ""]
                    if text:
                        out += notes_block(text) + [""]
            out.append("")
        with open(os.path.join(dest, "manifest.md"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(out))


NODE_CLASS_MAPPINGS = {
    "ToolsShotPaths": ShotPaths,
    "ToolsShotFinalize": ShotFinalize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToolsShotPaths": "Shot Paths (tools)",
    "ToolsShotFinalize": "Shot Finalize (tools)",
}
