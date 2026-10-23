"""The four-step endpoints (GOALS §6): data -> review -> train -> test."""

import json
import time

import pytest
from PIL import Image

from vlmforge.server import pipeline as pl
from vlmforge.server.workspace import Workspace

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vlmforge.server.api import create_app  # noqa: E402


class StubEngine:
    def status(self):
        return {"loaded": True, "current": "base", "pending": 0}

    def infer(self, images, skill, prompt, model="base"):
        if skill == "detect":
            return [{"boxes": [[0.25, 0.25, 0.75, 0.75]]} for _ in images]
        if skill == "point":
            return [{"points": [[0.5, 0.5]]} for _ in images]
        return [{"text": "2"} for _ in images]

    def teacher_generate(self, path, focus="", topics=None):
        return {
            "objects": [{"name": "person", "count": 1, "boxes": [[0.25, 0.25, 0.75, 0.75]]}],
            "questions": [{"q": "How many people are there?", "a": "2"}],
            "caption": "A person.",
            "model": "stub",
        }

    def teacher_generate_batch(self, paths, focus="", topics=None):
        return [self.teacher_generate(p, focus, topics) for p in paths]

    def judge_items(self, items):
        return [{"verdict": "accept", "note": "stub", "model": "stub"} for _ in items]


@pytest.fixture
def client(tmp_path, monkeypatch):
    from vlmforge import paths

    for k in ("WORKSPACE", "RUNS", "BASELINES", "ADAPTERS", "MANIFESTS"):
        monkeypatch.setattr(paths, k, tmp_path / k.lower())
    monkeypatch.setattr(paths, "BASELINES", tmp_path / "runs" / "baselines")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    folder = tmp_path / "cctv-1025"
    folder.mkdir()
    for c in ("red", "green", "blue"):
        Image.new("RGB", (64, 48), c).save(folder / f"{c}.jpg")
    ws = Workspace(tmp_path / "workspace")
    c = TestClient(create_app(ws, StubEngine()))
    c.ws, c.folder = ws, folder
    return c


def wait(client, qid):
    for _ in range(200):
        q = client.get(f"/api/queues/{qid}").json()
        if q["job"]["state"] in ("done", "error"):
            return q
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_data_review_train_test(client, monkeypatch):
    from vlmforge.server import workspace as wsmod

    monkeypatch.setattr(wsmod, "SPOT_MIN", 1)
    # ① a folder becomes a source; autogen runs on it
    r = client.post("/api/data/scan", json={"dir": str(client.folder)}).json()
    assert r == {"source": "cctv-1025", "added": 3, "found": 3, "image_ids": r["image_ids"]}
    q = client.post("/api/data/autogen", json={"source": "cctv-1025"}).json()
    q = wait(client, q["id"])
    assert q["job"]["state"] == "done" and q["total"] == 12  # 4 items x 3 images
    s = client.get("/api/review/summary").json()
    assert s["images"] == 3 and s["images_pending"] == 1 and s["auto"] == 11  # 1 spot item
    # ② review the spot image as a whole
    nx = client.get("/api/review/next-image", params={"reviewer": "P"}).json()
    assert len(nx["items"]) == 4
    client.post(
        "/api/review/image",
        json={
            "reviewer": "P",
            "items": [{"id": it["id"], "action": "approve"} for it in nx["items"]],
        },
    )
    assert client.get("/api/review/next-image", params={"reviewer": "P"}).json() == {}
    assert client.get("/api/review/images", params={"status": "reviewed"}).json()[0]["n"] == 4
    assert client.get("/api/review/summary").json()["trainable"] == 12
    # ③ train: the pipeline writes a config and would run the trainer; stub the run
    launched = {}

    def fake_run(self, name, st, cfg):
        launched["cfg"] = json.loads(json.dumps(__import__("yaml").safe_load(open(cfg))))
        st["state"], st["phase"] = "done", "done"
        self._write(name, st)

    monkeypatch.setattr(pl.Pipeline, "_run", fake_run)
    st = client.post("/api/train", json={"name": "ft-test", "steps": 40, "lr": 1e-4}).json()
    assert st["rows"] == 12 and st["images"] == 3
    for _ in range(50):
        st = client.get("/api/train/ft-test").json()
        if st["state"] == "done":
            break
        time.sleep(0.02)
    assert st["state"] == "done"
    cfg = launched["cfg"]
    assert cfg["optim"]["steps"] == 40 and cfg["optim"]["lr"] == 1e-4 and cfg["rank"] == 16
    assert cfg["data"]["train"][0]["path"].endswith("ft-test.jsonl")
    assert client.post("/api/train", json={"name": "ft-test"}).status_code == 409  # exists
    # ④ summary lists the run
    t = client.get("/api/test/summary").json()
    assert [r["name"] for r in t["runs"]] == ["ft-test"]
    assert client.get("/api/train").json()["trainable"] == 12
