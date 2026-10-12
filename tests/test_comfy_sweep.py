import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comfy_sweep  # noqa: E402
from comfy_sweep import SweepError, apply_values, build_runs, resolve_key  # noqa: E402

WORKFLOW = {
    "3": {"class_type": "KSampler", "_meta": {"title": "Main Sampler"},
          "inputs": {"seed": 0, "cfg": 7.0, "steps": 20, "model": ["4", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
    "6": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive"},
          "inputs": {"text": "a cat", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "_meta": {"title": "Negative"},
          "inputs": {"text": "", "clip": ["4", 1]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["3", 0]}},
}


def test_resolve_key_by_id_title_and_class_type():
    assert resolve_key(WORKFLOW, "3.cfg") == ("3", "cfg")
    assert resolve_key(WORKFLOW, "Main Sampler.cfg") == ("3", "cfg")
    assert resolve_key(WORKFLOW, "KSampler.cfg") == ("3", "cfg")
    assert resolve_key(WORKFLOW, "Positive.text") == ("6", "text")


@pytest.mark.parametrize("key, message", [
    ("CLIPTextEncode.text", "several nodes"),
    ("Nope.cfg", "no node"),
    ("KSampler.nope", "no input"),
    ("KSampler.model", "connected"),
    ("cfg", "<node>.<input>"),
])
def test_resolve_key_errors(key, message):
    with pytest.raises(SweepError, match=message):
        resolve_key(WORKFLOW, key)


def test_apply_values_copies():
    result = apply_values(WORKFLOW, {"KSampler.cfg": 4, "Positive.text": "a dog"})
    assert result["3"]["inputs"]["cfg"] == 4
    assert result["6"]["inputs"]["text"] == "a dog"
    assert WORKFLOW["3"]["inputs"]["cfg"] == 7.0


def test_build_runs_product_runs_and_fixed():
    runs = build_runs({"KSampler.steps": 10},
                      {"KSampler.cfg": [4, 6], "KSampler.seed": [1, 2]},
                      [{"Positive.text": "a"}, {"Positive.text": "b"}])
    assert len(runs) == 8
    assert runs[0] == {"KSampler.steps": 10, "Positive.text": "a", "KSampler.cfg": 4, "KSampler.seed": 1}
    assert build_runs() == [{}]
    with pytest.raises(SweepError):
        build_runs(params={"KSampler.cfg": 4})


def test_parse_value():
    assert comfy_sweep.parse_value("[1, 2]") == [1, 2]
    assert comfy_sweep.parse_value("4.5") == 4.5
    assert comfy_sweep.parse_value("a cat") == "a cat"


def test_load_workflow_rejects_ui_format(tmp_path):
    path = tmp_path / "ui.json"
    path.write_text(json.dumps({"nodes": [], "links": []}))
    with pytest.raises(SweepError, match="Export \\(API\\)"):
        comfy_sweep.load_workflow(path)


class FakeComfy(BaseHTTPRequestHandler):
    """Runs a prompt instantly; cfg 0 fails, so errors can be tested."""

    history = {}
    prompts = []

    def log_message(self, *args):
        pass

    def _send(self, body, status=200, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/interrupt":
            return self._send({})
        prompt = payload["prompt"]
        prompt_id = f"p{len(self.prompts)}"
        self.prompts.append(prompt)
        if prompt["3"]["inputs"]["cfg"] == 0:
            status = {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_type": "KSampler", "exception_message": "boom"}]]}
            outputs = {}
        else:
            status = {"status_str": "success", "completed": True, "messages": []}
            outputs = {"9": {"images": [
                {"filename": f"{prompt_id}.png", "subfolder": "", "type": "output"}]}}
        self.history[prompt_id] = {"status": status, "outputs": outputs}
        self._send({"prompt_id": prompt_id, "number": 0, "node_errors": {}})

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/history/"):
            prompt_id = parsed.path.rsplit("/", 1)[1]
            entry = self.history.get(prompt_id)
            return self._send({prompt_id: entry} if entry else {})
        if parsed.path == "/view":
            name = urllib.parse.parse_qs(parsed.query)["filename"][0]
            return self._send(f"png:{name}".encode(), content_type="image/png")
        self._send({}, 404)


@pytest.fixture
def server():
    FakeComfy.history = {}
    FakeComfy.prompts = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_sweep_end_to_end(server, tmp_path):
    workflow_path = tmp_path / "wf.json"
    workflow_path.write_text(json.dumps(WORKFLOW))
    out = tmp_path / "out"

    code = comfy_sweep.main([str(workflow_path), "--server", server, "--out", str(out),
                             "--param", "KSampler.cfg=[0, 5]", "--param", "KSampler.seed=[1, 2]",
                             "--set", "Positive.text=a dog"])

    assert code == 1  # the two cfg=0 runs fail
    assert [p["3"]["inputs"]["seed"] for p in FakeComfy.prompts] == [1, 2, 1, 2]
    assert all(p["6"]["inputs"]["text"] == "a dog" for p in FakeComfy.prompts)

    records = json.loads((out / "results.json").read_text())
    assert [r["status"] for r in records] == ["error", "error", "ok", "ok"]
    assert records[0]["error"] == "KSampler: boom"
    downloaded = Path(records[2]["files"][0]["path"])
    assert downloaded.read_bytes() == b"png:p2.png"
    assert json.loads((downloaded.parent / "prompt.json").read_text())["3"]["inputs"]["cfg"] == 5
    assert (out / "results.csv").read_text().splitlines()[0].startswith(
        "index,name,Positive.text,KSampler.cfg,KSampler.seed,status")


def test_dry_run_and_bad_key_queue_nothing(server, tmp_path, capsys):
    workflow_path = tmp_path / "wf.json"
    workflow_path.write_text(json.dumps(WORKFLOW))

    assert comfy_sweep.main([str(workflow_path), "--server", server, "--dry-run",
                             "--param", "KSampler.cfg=[4, 6]"]) == 0
    assert "[2/2]" in capsys.readouterr().out
    assert comfy_sweep.main([str(workflow_path), "--server", server,
                             "--param", "KSampler.cfg=[4]", "--set", "KSampler.nope=1"]) == 2
    assert FakeComfy.prompts == []


def test_list(tmp_path, capsys):
    workflow_path = tmp_path / "wf.json"
    workflow_path.write_text(json.dumps(WORKFLOW))
    assert comfy_sweep.main([str(workflow_path), "--list"]) == 0
    listing = capsys.readouterr().out
    assert '[3] KSampler  "Main Sampler"' in listing
    assert "cfg = 7.0" in listing
    assert "model" not in listing
