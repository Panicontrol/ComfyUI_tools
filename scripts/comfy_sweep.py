#!/usr/bin/env python3
"""Run one ComfyUI workflow many times with different parameters.

Takes a workflow exported in API format (*Workflow → Export (API)*), builds every
combination of the parameters you vary, queues them one at a time on a ComfyUI
server, waits for each to finish and downloads its outputs. Every run is logged
to ``results.json`` / ``results.csv`` in the output folder, next to the exact
prompt that was sent.

Parameters are addressed as ``<node>.<input>``, where ``<node>`` is a node id
("3"), a node title ("Main Sampler") or a class type ("KSampler") — a title or
class type must match exactly one node.

    # what can be varied?
    python scripts/comfy_sweep.py workflow_api.json --list

    # 3 x 2 = 6 runs
    python scripts/comfy_sweep.py workflow_api.json \\
        --server http://10.0.0.2:8188 \\
        --param 'KSampler.cfg=[4, 6, 8]' --param 'KSampler.seed=[1, 2]' \\
        --set 'KSampler.steps=20'

    # the same from a file: {"set": {...}, "params": {...}, "runs": [{...}, ...]}
    python scripts/comfy_sweep.py workflow_api.json --grid sweep.json

Only the standard library is used, so it runs from any Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import copy
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_SERVER = "http://127.0.0.1:8188"


class SweepError(Exception):
    """A problem with the workflow, the parameters or the server."""


# --- workflow ---------------------------------------------------------------


def load_workflow(path):
    with open(path, encoding="utf-8") as handle:
        workflow = json.load(handle)
    if isinstance(workflow, dict) and "nodes" in workflow and "links" in workflow:
        raise SweepError(
            f"{path} is a UI workflow; export it with Workflow → Export (API) instead"
        )
    if not isinstance(workflow, dict) or not all(
        isinstance(node, dict) and "class_type" in node for node in workflow.values()
    ):
        raise SweepError(f"{path} is not an API-format workflow")
    return workflow


def node_title(node):
    return (node.get("_meta") or {}).get("title", "")


def resolve_node(workflow, ref):
    """Map a node id, title or class type to a node id."""
    if ref in workflow:
        return ref
    for field, getter in (("title", node_title), ("class type", lambda n: n["class_type"])):
        matches = [node_id for node_id, node in workflow.items() if getter(node) == ref]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SweepError(
                f"{ref!r} matches several nodes by {field} ({', '.join(matches)}); use a node id"
            )
    raise SweepError(f"no node with id, title or class type {ref!r}")


def resolve_key(workflow, key):
    """Split ``<node>.<input>`` and return ``(node_id, input_name)``."""
    node_ref, dot, input_name = key.rpartition(".")
    if not dot or not node_ref or not input_name:
        raise SweepError(f"parameter {key!r} must look like <node>.<input>")
    node_id = resolve_node(workflow, node_ref)
    inputs = workflow[node_id].get("inputs", {})
    if input_name not in inputs:
        known = ", ".join(sorted(inputs)) or "none"
        raise SweepError(f"node {node_id} has no input {input_name!r} (inputs: {known})")
    if is_link(inputs[input_name]):
        raise SweepError(
            f"{key} is connected to another node; set the value on the node that feeds it"
        )
    return node_id, input_name


def is_link(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def apply_values(workflow, values):
    """Return a copy of ``workflow`` with ``{key: value}`` written into it."""
    result = copy.deepcopy(workflow)
    for key, value in values.items():
        node_id, input_name = resolve_key(result, key)
        result[node_id]["inputs"][input_name] = value
    return result


def describe_workflow(workflow):
    """Lines listing every node and its settable (non-linked) inputs."""
    lines = []
    for node_id in sorted(workflow, key=lambda k: (len(k), k)):
        node = workflow[node_id]
        title = node_title(node)
        header = f"[{node_id}] {node['class_type']}"
        if title and title != node["class_type"]:
            header += f"  \"{title}\""
        lines.append(header)
        for name, value in node.get("inputs", {}).items():
            if not is_link(value):
                lines.append(f"    {name} = {json.dumps(value, ensure_ascii=False)}")
    return lines


# --- parameters -------------------------------------------------------------


def parse_value(text):
    """JSON when it parses (numbers, lists, true, "quoted"), the raw string otherwise."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_assignment(text, option):
    key, eq, value = text.partition("=")
    if not eq or not key.strip():
        raise SweepError(f"{option} {text!r} must look like <node>.<input>=<value>")
    return key.strip(), parse_value(value)


def build_runs(fixed=None, params=None, runs=None):
    """Every run as a ``{key: value}`` dict: ``runs`` x product of ``params``, plus ``fixed``."""
    fixed = fixed or {}
    params = params or {}
    for key, values in params.items():
        if not isinstance(values, list) or not values:
            raise SweepError(f"param {key!r} needs a non-empty list of values")
    keys = list(params)
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*(params[k] for k in keys))]
    bases = runs if runs else [{}]
    return [{**fixed, **base, **combo} for base in bases for combo in grid]


def load_grid(path):
    with open(path, encoding="utf-8") as handle:
        grid = json.load(handle)
    unknown = set(grid) - {"set", "params", "runs"}
    if unknown:
        raise SweepError(f"{path}: unknown keys {sorted(unknown)} (expected set, params, runs)")
    return grid.get("set", {}), grid.get("params", {}), grid.get("runs", [])


# --- server -----------------------------------------------------------------


class ComfyClient:
    def __init__(self, server, headers=None, request_timeout=30):
        self.server = server.rstrip("/")
        self.headers = headers or {}
        self.request_timeout = request_timeout

    def _request(self, path, data=None):
        body = None
        headers = dict(self.headers)
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.server + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise SweepError(f"{path}: HTTP {error.code}: {detail[:2000]}") from None
        except urllib.error.URLError as error:
            raise SweepError(f"cannot reach {self.server}: {error.reason}") from None

    def json(self, path, data=None):
        return json.loads(self._request(path, data) or b"null")

    def submit(self, prompt, client_id):
        reply = self.json("/prompt", {"prompt": prompt, "client_id": client_id})
        if reply.get("node_errors"):
            raise SweepError(f"rejected: {json.dumps(reply['node_errors'])[:2000]}")
        return reply["prompt_id"]

    def wait(self, prompt_id, timeout, poll=1.0):
        """The history entry once the run has finished, ``None`` after ``timeout`` seconds."""
        # ComfyUI writes a history entry only when a prompt is done, failed or not.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            entry = (self.json(f"/history/{prompt_id}") or {}).get(prompt_id)
            if entry:
                return entry
            time.sleep(poll)
        return None

    def interrupt(self):
        try:
            self.json("/interrupt", {})
        except SweepError:
            pass

    def download(self, item, target):
        query = urllib.parse.urlencode(
            {
                "filename": item["filename"],
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output"),
            }
        )
        target.write_bytes(self._request(f"/view?{query}"))


def execution_error(entry):
    """The error message from a history entry, or ``None`` if the run succeeded."""
    status = entry.get("status") or {}
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            node = data.get("node_type") or data.get("node_id")
            return f"{node}: {data.get('exception_message', '').strip()}"
        if kind == "execution_interrupted":
            return "interrupted"
    if status.get("status_str") == "error":
        return "error"
    return None


def output_files(entry):
    """``(node_id, item)`` for every file a run produced (images, gifs, videos, audio…)."""
    for node_id, outputs in (entry.get("outputs") or {}).items():
        for items in outputs.values():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "filename" in item:
                        yield node_id, item


# --- sweep ------------------------------------------------------------------


def slug(values, limit=80):
    parts = []
    for key, value in values.items():
        text = value if isinstance(value, str) else json.dumps(value)
        parts.append(f"{key.rpartition('.')[2]}-{text}")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(parts)).strip("_")
    return name[:limit] or "run"


def run_sweep(client, workflow, runs, out_dir, timeout, download=True, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    client_id = uuid.uuid4().hex
    width = len(str(len(runs)))
    records = []
    for index, values in enumerate(runs, 1):
        name = f"{index:0{width}d}_{slug(values)}"
        run_dir = out_dir / name
        run_dir.mkdir(exist_ok=True)
        prompt = apply_values(workflow, values)
        (run_dir / "prompt.json").write_text(json.dumps(prompt, indent=2, ensure_ascii=False))
        record = {"index": index, "name": name, "params": values, "status": "ok",
                  "error": None, "seconds": None, "prompt_id": None, "files": []}
        log(f"[{index}/{len(runs)}] {json.dumps(values, ensure_ascii=False)}")
        started = time.monotonic()
        try:
            record["prompt_id"] = client.submit(prompt, client_id)
            entry = client.wait(record["prompt_id"], timeout)
            record["seconds"] = round(time.monotonic() - started, 2)
            if entry is None:
                client.interrupt()
                record.update(status="timeout", error=f"no result after {timeout}s")
            else:
                error = execution_error(entry)
                if error:
                    record.update(status="error", error=error)
                for node_id, item in output_files(entry):
                    file_info = {"node": node_id, **item}
                    if download:
                        target = run_dir / f"{node_id}_{Path(item['filename']).name}"
                        client.download(item, target)
                        file_info["path"] = str(target)
                    record["files"].append(file_info)
        except SweepError as error:
            record.update(status="error", error=str(error))
            if record["seconds"] is None:
                record["seconds"] = round(time.monotonic() - started, 2)
        log(f"    {record['status']} in {record['seconds']}s, {len(record['files'])} file(s)"
            + (f" — {record['error']}" if record["error"] else ""))
        records.append(record)
        write_results(out_dir, records)
    return records


def write_results(out_dir, records):
    (out_dir / "results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    keys = list(dict.fromkeys(k for r in records for k in r["params"]))
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "name", *keys, "status", "seconds", "error", "files"])
        for r in records:
            files = ";".join(f.get("path", f["filename"]) for f in r["files"])
            writer.writerow([r["index"], r["name"],
                             *(json.dumps(r["params"].get(k), ensure_ascii=False) for k in keys),
                             r["status"], r["seconds"], r["error"] or "", files])


# --- command line -----------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a ComfyUI API workflow over a grid of parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument("workflow", help="workflow in API format (Workflow → Export (API))")
    parser.add_argument("--server", default=os.environ.get("COMFYUI_URL", DEFAULT_SERVER),
                        help=f"ComfyUI address (env COMFYUI_URL, default {DEFAULT_SERVER})")
    parser.add_argument("--grid", help="JSON file with set / params / runs")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=[V1,V2,...]",
                        help="vary an input over a JSON list of values (repeatable)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="fix an input for every run (repeatable)")
    parser.add_argument("--out", help="output folder (default sweeps/<workflow>_<timestamp>)")
    parser.add_argument("--timeout", type=float, default=1800,
                        help="seconds to wait for one run before interrupting it (default 1800)")
    parser.add_argument("--header", action="append", default=[], metavar="NAME: VALUE",
                        help="extra HTTP header, e.g. 'Authorization: Bearer …' (repeatable)")
    parser.add_argument("--no-download", action="store_true",
                        help="record output file names without downloading them")
    parser.add_argument("--list", action="store_true",
                        help="list the workflow's nodes and settable inputs, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the parameters and print the runs without queuing them")
    args = parser.parse_args(argv)

    try:
        workflow = load_workflow(args.workflow)
        if args.list:
            print("\n".join(describe_workflow(workflow)))
            return 0

        fixed, params, runs = load_grid(args.grid) if args.grid else ({}, {}, [])
        fixed = {**fixed, **dict(parse_assignment(s, "--set") for s in args.set)}
        params = {**params, **dict(parse_assignment(p, "--param") for p in args.param)}
        all_runs = build_runs(fixed, params, runs)
        for values in all_runs:
            apply_values(workflow, values)  # fail before queuing anything

        if args.dry_run:
            for index, values in enumerate(all_runs, 1):
                print(f"[{index}/{len(all_runs)}] {json.dumps(values, ensure_ascii=False)}")
            return 0

        headers = {}
        for header in args.header:
            name, colon, value = header.partition(":")
            if not colon:
                raise SweepError(f"--header {header!r} must look like 'Name: value'")
            headers[name.strip()] = value.strip()

        out_dir = Path(args.out) if args.out else (
            Path("sweeps") / f"{Path(args.workflow).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        client = ComfyClient(args.server, headers)
        print(f"{len(all_runs)} run(s) on {client.server} → {out_dir}")
        records = run_sweep(client, workflow, all_runs, out_dir, args.timeout,
                            download=not args.no_download)
    except SweepError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    failed = sum(r["status"] != "ok" for r in records)
    print(f"done: {len(records) - failed} ok, {failed} failed — {out_dir / 'results.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
