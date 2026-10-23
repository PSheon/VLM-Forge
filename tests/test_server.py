import io
import time

import pytest
from PIL import Image

from vlmforge.server.workspace import Workspace, item_row

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vlmforge.server.api import create_app  # noqa: E402


class StubEngine:
    """Deterministic predictions: one box / point in the middle, canned text."""

    def __init__(self):
        self.calls = []

    def status(self):
        return {"loaded": True, "current": "base", "pending": 0}

    def infer(self, images, skill, prompt, model="base"):
        self.calls.append((len(images), skill, prompt, model))
        if skill == "detect":
            return [{"boxes": [[0.25, 0.25, 0.75, 0.75]]} for _ in images]
        if skill == "point":
            return [{"points": [[0.5, 0.5]]} for _ in images]
        return [{"text": f"{model}: 2"} for _ in images]

    def judge_items(self, items):
        # "shelf" items are flagged, everything else accepted
        return [
            {
                "verdict": "fix" if it["prompt"] == "shelf" else "accept",
                "note": "stub",
                "model": "stub",
            }
            for it in items
        ]


def png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from vlmforge import paths

    monkeypatch.setattr(paths, "WORKSPACE", tmp_path / "ws")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "BASELINES", tmp_path / "runs" / "baselines")
    monkeypatch.setattr(paths, "ADAPTERS", tmp_path / "adapters")
    ws = Workspace(tmp_path / "ws")
    eng = StubEngine()
    c = TestClient(create_app(ws, eng))
    c.ws, c.eng = ws, eng
    return c


def test_workbench_flow(client):
    im = client.post("/api/images", files={"file": ("a.png", png_bytes("red"), "image/png")}).json()
    assert client.get(f"/api/images/{im['id']}/file").status_code == 200
    r = client.post(
        "/api/infer", json={"image_id": im["id"], "skill": "detect", "prompt": "person"}
    ).json()
    assert r["pred"]["boxes"] == [[0.25, 0.25, 0.75, 0.75]]
    saved = client.post(
        "/api/items",
        json={
            "image_id": im["id"],
            "skill": "detect",
            "prompt": "person",
            "pred": r["pred"],
            "pred_model": "base",
            "action": "correct",
            "label": {"boxes": [[0.2, 0.2, 0.7, 0.7]]},
            "reviewer": "Paul",
        },
    ).json()
    assert saved["status"] == "corrected" and saved["origin"] == "workbench"
    rows = client.ws.reviewed_rows()
    assert rows[0]["boxes"] == [[0.2, 0.2, 0.7, 0.7]] and rows[0]["changed"] is True
    assert rows[0]["label_origin"] == "human" and rows[0]["prefill"] == "base"


def test_queue_claim_review_export(client):
    ids = [
        client.post("/api/images", files={"file": (f"{c}.png", png_bytes(c), "image/png")}).json()[
            "id"
        ]
        for c in ("red", "green", "blue")
    ]
    q = client.post(
        "/api/queues",
        json={
            "name": "cctv-1",
            "skill": "detect",
            "prompts": ["person", "shelf"],
            "image_ids": ids,
            "judge": False,
        },
    ).json()
    assert q["total"] == 6
    for _ in range(50):  # background predictions
        job = client.get(f"/api/queues/{q['id']}").json()["job"]
        if job["state"] == "done":
            break
        time.sleep(0.02)
    assert job["state"] == "done" and job["done"] == 6
    # two reviewers never get the same item
    a = client.get(f"/api/queues/{q['id']}/next", params={"reviewer": "A"}).json()
    b = client.get(f"/api/queues/{q['id']}/next", params={"reviewer": "B"}).json()
    assert a["id"] != b["id"] and a["pred"]["boxes"]
    # the same reviewer asking again gets their own claim back
    assert (
        client.get(f"/api/queues/{q['id']}/next", params={"reviewer": "A"}).json()["id"] == a["id"]
    )
    client.post(f"/api/items/{a['id']}/review", json={"action": "approve", "reviewer": "A"})
    client.post(
        f"/api/items/{b['id']}/review",
        json={"action": "correct", "label": {"boxes": []}, "reviewer": "B"},
    )
    nxt = client.get(f"/api/queues/{q['id']}/next", params={"reviewer": "A"}).json()
    assert nxt["id"] not in (a["id"], b["id"])
    client.post(f"/api/items/{nxt['id']}/review", json={"action": "discard", "reviewer": "A"})
    counts = client.get(f"/api/queues/{q['id']}").json()["counts"]
    assert counts == {"pending": 3, "auto": 0, "approved": 1, "corrected": 1, "discarded": 1}
    ex = client.post("/api/export", json={"name": "cctv-1", "incoming": True}).json()
    assert ex["train"]["rows"] == 2 and ex["train"]["skills"] == {"detect": 2}
    assert ex["incoming"]["items"] == 2 and ex["incoming"]["images"] == 1  # a, b: same image
    bad = client.post(f"/api/items/{a['id']}/review", json={"action": "correct", "reviewer": "A"})
    assert bad.status_code == 422


def test_judge_triage(client, monkeypatch):
    from vlmforge.server import workspace as wsmod

    monkeypatch.setattr(wsmod, "SPOT_MIN", 1)
    ids = [
        client.post("/api/images", files={"file": (f"{c}.png", png_bytes(c), "image/png")}).json()[
            "id"
        ]
        for c in ("red", "green", "blue", "white")
    ]
    q = client.post(
        "/api/queues",
        json={"name": "j", "skill": "detect", "prompts": ["person", "shelf"], "image_ids": ids},
    ).json()
    for _ in range(100):
        q = client.get(f"/api/queues/{q['id']}").json()
        if q["job"]["state"] == "done":
            break
        time.sleep(0.02)
    assert q["job"]["state"] == "done", q["job"]
    # 4 person items accepted: 1 spot-checked stays pending, 3 go auto; 4 shelf items flagged
    assert q["verdicts"] == {"accept": 4, "fix": 4} and q["spot"] == 1
    assert q["counts"]["auto"] == 3 and q["counts"]["pending"] == 5
    # flagged items come first for the human, then the spot sample
    first = client.get(f"/api/queues/{q['id']}/next", params={"reviewer": "A"}).json()
    assert first["prompt"] == "shelf" and first["judge"]["verdict"] == "fix"
    # export: auto items are machine_reviewed with the prediction as label
    rows = client.ws.reviewed_rows([q["id"]])
    assert len(rows) == 3 and {r["label_origin"] for r in rows} == {"machine_reviewed"}
    assert rows[0]["boxes"] == [[0.25, 0.25, 0.75, 0.75]] and rows[0]["reviewer"] == "judge:stub"
    # a single judge call from the workbench
    v = client.post(
        "/api/judge",
        json={"image_id": ids[0], "skill": "detect", "prompt": "shelf", "pred": {"boxes": []}},
    ).json()
    assert v["verdict"] == "fix"


def test_compare_and_status(client):
    im = client.post("/api/images", files={"file": ("a.png", png_bytes("red"), "image/png")}).json()
    r = client.post(
        "/api/compare",
        json={
            "image_id": im["id"],
            "skill": "query",
            "prompt": "How many?",
            "models": ["base", "x.pt"],
        },
    ).json()
    assert r["base"]["pred"]["text"] == "base: 2" and r["x.pt"]["pred"]["text"] == "x.pt: 2"
    st = client.get("/api/status").json()
    assert st["engine"]["loaded"] and st["models"][0]["name"] == "base"


def test_item_row_shapes():
    base = {
        "path": "/p.jpg",
        "source": "upload",
        "id": 1,
        "reviewer": "P",
        "reviewed_at": "2026-10-23T00:00:00",
        "pred_model": "base",
        "status": "approved",
    }
    assert (
        item_row({**base, "skill": "query", "prompt": "q?", "label": {"text": "3"}})["answer"]
        == "3"
    )
    assert (
        item_row({**base, "skill": "caption", "prompt": "short", "label": {"text": "x"}})["length"]
        == "short"
    )
    assert item_row({**base, "skill": "point", "prompt": "p", "label": {"points": [[0.1, 0.2]]}})[
        "points"
    ]
