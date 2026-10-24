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

    def teacher_generate_batch(self, paths, focus="", topics=None):
        return [
            {
                "objects": [{"name": "person", "boxes": [[0.25, 0.25, 0.75, 0.75]]}],
                "questions": [{"q": "How many people are there?", "a": "2"}],
                "caption": "A person in a room.",
                "model": "stub",
            }
            for _ in paths
        ]

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
    c.tmp_heldout = tmp_path / "heldout-cctv"
    c.tmp_heldout.mkdir()
    return c


def wait_done(client, qid):
    for _ in range(200):
        q = client.get(f"/api/queues/{qid}").json()
        if q["job"]["state"] in ("done", "error"):
            return q
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_review_claims_and_whole_image_decisions(client):
    """A disagreeing student sends questions to humans per image; two reviewers never hold
    the same image; one image's decisions are saved together; held-out folders are refused."""
    client.eng.infer = lambda images, skill, prompt, model="base": [
        {"boxes": []}
        if skill == "detect"
        else {"points": []}
        if skill == "point"
        else {"text": "5"}
        for _ in images
    ]
    ds = client.post("/api/datasets", json={"name": "d"}).json()
    for c in ("red", "green"):
        client.post(
            f"/api/datasets/{ds['id']}/upload",
            files={"file": (f"{c}.png", png_bytes(c), "image/png")},
        )
    q = wait_done(client, client.post(f"/api/datasets/{ds['id']}/label").json()["id"])
    assert q["job"]["state"] == "done", q["job"]
    a = client.get("/api/review/next-image", params={"reviewer": "A", "dataset_id": ds["id"]})
    b = client.get("/api/review/next-image", params={"reviewer": "B", "dataset_id": ds["id"]})
    a, b = a.json(), b.json()
    assert a["image"]["id"] != b["image"]["id"]  # each claim locks a whole image
    pending = [it for it in a["items"] if it["status"] == "pending"]
    fixes = [it for it in a["items"] if it["verdict"] == "fix"]
    assert fixes and all(it["status"] == "pending" for it in fixes)  # every disagreement → human
    out = client.post(
        "/api/review/image",
        json={
            "reviewer": "A",
            "items": [
                {"id": it["id"], "action": "correct", "label": it["teacher"]} for it in pending
            ],
        },
    ).json()
    assert {o["status"] for o in out} == {"corrected"} and {o["reviewer"] for o in out} == {"A"}
    nxt = client.get("/api/review/next-image", params={"reviewer": "A", "dataset_id": ds["id"]})
    assert nxt.json() == {}  # the only image left is B's for the next 10 minutes
    bad = client.post(
        f"/api/items/{pending[0]['id']}/review", json={"action": "correct", "reviewer": "A"}
    )
    assert bad.status_code == 422  # correct needs a label
    held = client.post(f"/api/datasets/{ds['id']}/folder", json={"dir": str(client.tmp_heldout)})
    assert held.status_code == 422


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


def test_retriage_keeps_the_spot_sample(tmp_path, monkeypatch):
    """Re-running triage (e.g. on 接著跑) over a half-reviewed batch must not
    re-draw the spot check: once it auto-passed 128 of 149 sampled questions."""
    from vlmforge.server import workspace as wsmod

    monkeypatch.setattr(wsmod, "SPOT_MIN", 0)
    monkeypatch.setattr(wsmod, "SPOT_FRAC", 0.2)
    from vlmforge.server.datasets import DatasetStore

    ws = Workspace(tmp_path / "ws")
    DatasetStore(ws)  # adds queues.dataset_id, as the app always does
    im = ws.add_image_bytes(png_bytes("red"), ext="png")
    q = ws.create_queue("b", "query", "base")
    box = {"boxes": [[0.1, 0.1, 0.5, 0.5]]}
    for i in range(50):
        iid = ws.add_item(im["id"], "detect", f"o{i}", box, "base", q["id"], "auto", box)
        ws.set_judge(iid, {"verdict": "accept", "note": "", "model": "agreement"})
    first = ws.triage_items(q["id"])
    assert first["spot"] == 10 and first["auto"] == 40
    sampled = {it["id"] for it in ws.items(q["id"], "pending", wsmod.ALL)}
    # a few more accepted items arrive (e.g. re-answered questions), then triage runs again
    for i in range(5):
        iid = ws.add_item(im["id"], "detect", f"n{i}", box, "base", q["id"], "auto", box)
        ws.set_judge(iid, {"verdict": "accept", "note": "", "model": "agreement"})
    second = ws.triage_items(q["id"])
    still = {it["id"] for it in ws.items(q["id"], "pending", wsmod.ALL)}
    assert sampled <= still  # nothing already drawn was auto-passed
    assert second["auto"] == 4 and len(still) == 11  # budget round(0.2 * 55) = 11 → 1 new draw
    # 只看抽查: an image whose only pending items are not spot-checks is skipped
    im2 = ws.add_image_bytes(png_bytes("blue"), ext="png")
    fix = ws.add_item(im2["id"], "detect", "x", box, "base", q["id"], "auto", box)
    ws.set_judge(fix, {"verdict": "fix", "note": "", "model": "agreement"})
    image, _ = ws.claim_next_image(q["id"], "P", only_spot=True)
    assert image["id"] == im["id"]
    image, _ = ws.claim_next_image(q["id"], "Q")
    assert image["id"] == im2["id"]  # without the filter the disagreement comes first
    # Ctrl+Z on a saved spot-check question: back to pending with its auto label, not empty
    sid = next(iter(sampled))
    auto = ws.item(sid)["label"]
    ws.review(sid, "correct", "P", {"boxes": []})
    back = ws.reopen(sid)
    assert back["status"] == "pending" and back["label"] == auto and back["reviewer"] is None
