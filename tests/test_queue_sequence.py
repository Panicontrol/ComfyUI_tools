import csv
import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "queue_sequence.py")
spec = importlib.util.spec_from_file_location("queue_sequence", SCRIPT)
queue_sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue_sequence)


class FakeComfy:
    """/prompt runs the graph at once (writes the done file, fails where told), /history answers."""

    def __init__(self, done_pattern, fail_frames=()):
        self.prompts, self.history, self.deleted = [], {}, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/prompt":
                    graph = payload["prompt"]
                    prompt_id = f"p{len(fake.prompts)}"
                    fake.prompts.append(json.loads(json.dumps(graph)))
                    frame = graph["30"]["inputs"]["start_frame_index"]
                    ok = frame not in fail_frames
                    if ok:
                        open(done_pattern.format(frame=frame), "w").close()
                    messages = [["execution_start", {"timestamp": 1000}],
                                ["execution_success" if ok else "execution_error",
                                 {"timestamp": 3500, "exception_type": "OOM", "exception_message": "out of memory"}]]
                    fake.history[prompt_id] = {"status": {"status_str": "success" if ok else "error",
                                                          "completed": ok, "messages": messages}}
                    self.reply({"prompt_id": prompt_id, "node_errors": {}})
                else:
                    fake.deleted += payload.get("delete", [])
                    self.reply({})

            def do_GET(self):
                prompt_id = self.path.rsplit("/", 1)[-1]
                self.reply({prompt_id: fake.history[prompt_id]} if prompt_id in fake.history else {})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def test_queue_sequence_skips_done_frames_logs_and_resumes(tmp_path):
    frames, out = tmp_path / "frames", tmp_path / "out"
    frames.mkdir(); out.mkdir()
    for n in range(1001, 1006):
        (frames / f"seq01.{n:05d}.png").write_bytes(b"")
    done = str(out / "up.{frame:05d}.png")
    open(done.format(frame=1002), "w").close()                      # finished in an earlier run
    graph = {"12": {"class_type": "ToolsLoadImageSequence",
                    "inputs": {"directory": str(frames), "pattern": "*.png", "sort_by": "name", "reverse": False,
                               "start_index": 0, "step": 1, "count": 1, "on_size_mismatch": "error", "index": 0}},
             "30": {"class_type": "ToolsSaveImageSequence", "inputs": {"start_frame_index": 1001}}}
    workflow = tmp_path / "graph_api.json"
    workflow.write_text(json.dumps(graph))
    log = tmp_path / "times.csv"

    comfy = FakeComfy(done, fail_frames={1004})
    args = [str(workflow), "--load-node", "12", "--save-node", "30", "--done", done, "--server", comfy.url,
            "--log", str(log)]
    with pytest.raises(SystemExit, match="stopped at index 3"):
        queue_sequence.main(args)
    sent = [(p["12"]["inputs"]["index"], p["30"]["inputs"]["start_frame_index"]) for p in comfy.prompts]
    assert sent[:3] == [(0, 1001), (2, 1003), (3, 1004)]           # index 1 (frame 1002) skipped

    comfy2 = FakeComfy(done)                                        # the same command again: carries on
    assert queue_sequence.main(args[:-4] + ["--server", comfy2.url, "--log", str(log)]) == 0
    # index 4 was queued ahead while 3 ran (this fake runs a graph the moment it is queued): done, skipped
    assert [p["12"]["inputs"]["index"] for p in comfy2.prompts] == [3]
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    assert [(r["index"], r["status"]) for r in rows] == [("0", "success"), ("2", "success"), ("3", "error"),
                                                         ("3", "success")]
    assert rows[0]["seconds"] == "2.5" and "OOM: out of memory" in [r["error"] for r in rows][2]
    assert "p0" in comfy.deleted and all(os.path.exists(done.format(frame=f)) for f in range(1001, 1006))


def test_queue_sequence_wants_the_api_format_and_a_count(tmp_path):
    ui = tmp_path / "ui.json"
    ui.write_text(json.dumps({"nodes": [], "links": []}))
    with pytest.raises(SystemExit, match="Export \\(API\\)"):
        queue_sequence.main([str(ui), "--load-node", "12"])
    with pytest.raises(SystemExit, match="count to the frames per run"):
        queue_sequence.sequence_frames({"directory": str(tmp_path), "count": 0})
