"""Run a ComfyUI graph once per frame of a long sequence, through ComfyUI's API.

The graph loads its frames with Load Image Sequence (tools) -- count 1 (or a
few), the index widget stepping through the sequence -- and writes them with
Save Image Sequence (tools), need_video off. This script queues the graph once
per index, so a sequence too long to hold in memory (5400 frames of 7204x2304
is ~1 TB as float32) goes through one piece at a time, with the models staying
loaded between runs.

    python queue_sequence.py graph_api.json --load-node 12 \\
        --done "D:/out/seq01_up/seq01_up_v001/seq01_up_v001.{frame:05d}.png" \\
        --server http://127.0.0.1:8190 --log seq01_times.csv

graph_api.json: the graph exported in API format (Workflow > Export (API)).
Link Load Image Sequence's frame_number output to Save Image Sequence's
start_frame_index (convert the widget to an input), or pass --save-node so
the script sets it, and the output frames keep the source's numbers.

--done: the output file of one frame, {frame} being its source frame number
(e.g. 1001). Frames whose file exists are skipped, so after a crash the same
command carries on where it stopped (Save writes each frame under a temporary
name until it is complete). Unpack a video into frames first with ffmpeg:

    ffmpeg -i seq01.mp4 -start_number 1001 D:/frames/seq01/seq01.%05d.png

Only the standard library is used.
"""

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import time
import urllib.request
import uuid


def natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def sequence_frames(inputs):
    """[(index, frame number, file names)] the way Load Image Sequence (tools) slices the folder."""
    directory = os.path.expanduser(str(inputs["directory"]).strip().strip('"'))
    globs = [p.strip() for p in str(inputs.get("pattern", "*.png")).split(",") if p.strip()] or ["*"]
    names = [n for n in os.listdir(directory) if os.path.isfile(os.path.join(directory, n))
             and any(fnmatch.fnmatch(n.lower(), g.lower()) for g in globs)]
    sort_by = inputs.get("sort_by", "name")
    if sort_by == "modified":
        names.sort(key=lambda n: os.path.getmtime(os.path.join(directory, n)))
    elif sort_by == "size":
        names.sort(key=lambda n: os.path.getsize(os.path.join(directory, n)))
    else:
        names.sort(key=natural_key)
    if inputs.get("reverse"):
        names.reverse()
    positions = list(range(len(names)))[int(inputs.get("start_index", 0))::int(inputs.get("step", 1))]
    count = int(inputs.get("count", 0))
    if count < 1:
        raise SystemExit("set the Load Image Sequence node's count to the frames per run (1 for one frame)")
    pieces = []
    for index in range(-(-len(positions) // count)):
        chosen = [names[p] for p in positions[index * count:(index + 1) * count]]
        numbers = re.findall(r"\d+", os.path.splitext(chosen[0])[0])
        pieces.append((index, int(numbers[-1]) if numbers else positions[index * count], chosen))
    return pieces


def request(server, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(server.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as response:
        body = response.read()
    return json.loads(body) if body else {}


def run_seconds(entry):
    """execution_start -> execution_success/error from ComfyUI's history, in seconds."""
    stamps = {kind: info.get("timestamp") for kind, info in entry.get("status", {}).get("messages", [])}
    start = stamps.get("execution_start")
    end = stamps.get("execution_success") or stamps.get("execution_error") or stamps.get("execution_interrupted")
    return round((end - start) / 1000.0, 2) if start and end else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workflow", help="the graph in API format (Export (API))")
    parser.add_argument("--load-node", required=True, help="id of the Load Image Sequence (tools) node")
    parser.add_argument("--save-node", help="id of the Save Image Sequence (tools) node: set its start_frame_index")
    parser.add_argument("--done", help="output file of one frame, {frame} = source frame number; existing = skip")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--start", type=int, default=0, help="first index")
    parser.add_argument("--end", type=int, help="last index (included); default: the end of the sequence")
    parser.add_argument("--ahead", type=int, default=1, help="runs queued ahead of the one running")
    parser.add_argument("--log", default="queue_sequence.csv", help="CSV of seconds per run")
    parser.add_argument("--keep-going", action="store_true", help="go on after a failed run")
    args = parser.parse_args(argv)

    with open(args.workflow, encoding="utf-8") as handle:
        graph = json.load(handle)
    if "nodes" in graph:
        raise SystemExit("this is the UI format: export the graph with Workflow > Export (API)")
    load = graph.get(str(args.load_node))
    if not load or "directory" not in load.get("inputs", {}):
        raise SystemExit(f"node {args.load_node} is not a Load Image Sequence (tools) node in {args.workflow}")
    pieces = sequence_frames(load["inputs"])
    last = len(pieces) - 1 if args.end is None else min(args.end, len(pieces) - 1)
    todo = [p for p in pieces if args.start <= p[0] <= last]
    skipped = [p for p in todo if args.done and os.path.exists(args.done.format(frame=p[1]))]
    todo = [p for p in todo if p not in skipped]
    print(f"{len(pieces)} pieces in the sequence; {len(skipped)} already done, {len(todo)} to run")

    client = str(uuid.uuid4())
    new_log = not os.path.exists(args.log)
    failed = 0
    with open(args.log, "a", newline="", encoding="utf-8") as handle:
        log = csv.writer(handle)
        if new_log:
            log.writerow(["index", "frame", "prompt_id", "seconds", "status", "error"])
        waiting = []  # (index, frame, prompt id, queued at)
        queue = list(todo)
        while queue or waiting:
            while queue and len(waiting) <= args.ahead:
                index, frame, _ = queue.pop(0)
                load["inputs"]["index"] = index
                if args.save_node:
                    graph[str(args.save_node)]["inputs"]["start_frame_index"] = frame
                answer = request(args.server, "/prompt", {"prompt": graph, "client_id": client})
                if answer.get("node_errors"):
                    raise SystemExit(f"ComfyUI refused the graph: {json.dumps(answer['node_errors'])[:800]}")
                waiting.append((index, frame, answer["prompt_id"], time.time()))
            index, frame, prompt_id, queued = waiting[0]
            entry = request(args.server, f"/history/{prompt_id}").get(prompt_id)
            if not entry or not entry.get("status", {}).get("completed", False) and \
                    entry.get("status", {}).get("status_str") != "error":
                time.sleep(0.5)
                continue
            waiting.pop(0)
            status = entry["status"].get("status_str", "?")
            error = ""
            if status != "success":
                failed += 1
                for kind, info in entry["status"].get("messages", []):
                    if kind == "execution_error":
                        error = f"{info.get('exception_type', '')}: {info.get('exception_message', '')}".strip()
            seconds = run_seconds(entry) or round(time.time() - queued, 2)
            log.writerow([index, frame, prompt_id, seconds, status, error])
            handle.flush()
            print(f"index {index} (frame {frame}): {status}, {seconds} s" + (f" -- {error}" if error else ""))
            try:  # keep ComfyUI's history from growing over thousands of runs
                request(args.server, "/history", {"delete": [prompt_id]})
            except Exception:
                pass
            if failed and not args.keep_going:
                for _, _, pending, _ in waiting:
                    try:
                        request(args.server, "/queue", {"delete": [pending]})
                    except Exception:
                        pass
                raise SystemExit(f"stopped at index {index}; run the same command again to carry on")
    print(f"done: {len(todo) - failed} runs, {failed} failed; times in {args.log}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
