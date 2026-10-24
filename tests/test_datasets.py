"""Datasets (GOALS §4): create, add images, fixed + auto labelling, splits, export, train."""

import io
import time
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from vlmforge.autogen import items_from_fixed, parse_fixed
from vlmforge.orin import BUNDLE_FILES
from vlmforge.server import pipeline as pl
from vlmforge.server.datasets import split_for
from vlmforge.server.workspace import Workspace

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vlmforge.server.api import create_app  # noqa: E402


class StubEngine:
    def status(self):
        return {"loaded": True, "current": "base", "pending": 0}

    def infer(self, images, skill, prompt, model="base"):
        if skill == "detect":  # a person is always found, a helmet never
            return [
                {"boxes": [] if prompt == "helmet" else [[0.25, 0.25, 0.75, 0.75]]} for _ in images
            ]
        if skill == "point":
            return [{"points": [[0.5, 0.5]]} for _ in images]
        return [{"text": "1" if prompt.startswith("How many") else "no"} for _ in images]

    def teacher_generate_batch(self, paths, focus="", topics=None):
        return [
            {
                "objects": [{"name": "person", "count": 1, "boxes": [[0.25, 0.25, 0.75, 0.75]]}],
                "questions": [{"q": "How many people are there?", "a": "1"}],
                "caption": "A person.",
                "model": "stub",
            }
            for _ in paths
        ]

    def teacher_fixed_batch(self, paths, fixed):
        return [
            {
                "objects": [
                    {"name": "person", "count": 1, "boxes": [[0.25, 0.25, 0.75, 0.75]]},
                    {"name": "helmet", "count": 0, "boxes": []},
                ],
                "questions": [{"q": q, "a": "no"} for q in fixed["query"]],
                "caption": "A person without a helmet.",
                "model": "stub",
            }
            for _ in paths
        ]

    def judge_items(self, items):
        return [{"verdict": "accept", "note": "stub", "model": "stub"} for _ in items]


def png(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from vlmforge import paths

    for k in ("WORKSPACE", "ADAPTERS", "MANIFESTS"):
        monkeypatch.setattr(paths, k, tmp_path / k.lower())
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "BASELINES", tmp_path / "runs" / "baselines")
    ws = Workspace(tmp_path / "workspace")
    c = TestClient(create_app(ws, StubEngine()))
    c.ws, c.tmp = ws, tmp_path
    return c


def wait(client, qid):
    for _ in range(200):
        q = client.get(f"/api/queues/{qid}").json()
        if q["job"]["state"] in ("done", "error"):
            return q
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_parse_fixed_and_items():
    f = parse_fixed(
        "detect: Person, helmet\nquery: Is anyone without a helmet?\n"
        "point: person\ncaption:\nbad line"
    )
    assert f == {
        "detect": ["person", "helmet"],
        "point": ["person"],
        "query": ["Is anyone without a helmet?"],
        "caption": ["normal"],
    }
    t = {
        "objects": [{"name": "person", "count": 1, "boxes": [[0, 0, 0.5, 0.5]]}],
        "questions": [{"q": "Is anyone without a helmet?", "a": "no"}],
        "caption": "x",
    }
    kinds = [(i["skill"], i["prompt"]) for i in items_from_fixed(t, f)]
    assert kinds == [
        ("detect", "person"),
        ("detect", "helmet"),  # asked but absent -> negative example
        ("point", "person"),
        ("query", "Is anyone without a helmet?"),
        ("caption", "normal"),
    ]


def test_split_is_deterministic_and_about_ten_percent():
    ids = [f"{i:08x}" + "0" * 56 for i in range(0, 4000, 4)]
    vals = sum(split_for(i) == "val" for i in ids)
    assert 60 <= vals <= 140 and split_for(ids[0]) == split_for(ids[0])


def test_dataset_flow(client, monkeypatch):
    from vlmforge.server import workspace as wsmod

    monkeypatch.setattr(wsmod, "SPOT_MIN", 0)
    monkeypatch.setattr(wsmod, "SPOT_FRAC", 0)  # no spot sample: every agreed item goes auto
    ds = client.post(
        "/api/datasets",
        json={
            "name": "工廠-1025",
            "config": {
                "mode": "both",
                "fixed": "detect: helmet\nquery: Is anyone without a helmet?",
            },
        },
    ).json()
    assert ds["images"] == 0
    # upload one image and a zip of two
    client.post(
        f"/api/datasets/{ds['id']}/upload", files={"file": ("a.png", png("red"), "image/png")}
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x/b.png", png("green"))
        z.writestr("c.png", png("blue"))
        z.writestr("notes.txt", b"skip")
    r = client.post(
        f"/api/datasets/{ds['id']}/upload",
        files={"file": ("batch.zip", buf.getvalue(), "application/zip")},
    ).json()
    assert r["added"] == 2
    folder = client.tmp / "frames"
    folder.mkdir()
    Image.new("RGB", (64, 48), "white").save(folder / "d.jpg")
    assert (
        client.post(f"/api/datasets/{ds['id']}/folder", json={"dir": str(folder)}).json()["added"]
        == 1
    )
    d = client.get(f"/api/datasets/{ds['id']}").json()
    assert d["images"] == 4 and d["unlabelled"] == 4 and len(d["thumbs"]) == 4
    assert client.post(f"/api/datasets/{ds['id']}/folder", json={"dir": "/nope"}).status_code == 404
    assert client.post("/api/datasets/999/folder", json={"dir": str(folder)}).status_code == 404
    # label: auto + fixed, deduplicated per (skill, prompt)
    q = client.post(f"/api/datasets/{ds['id']}/label").json()
    q = wait(client, q["id"])
    assert q["job"]["state"] == "done", q["job"]
    d = client.get(f"/api/datasets/{ds['id']}").json()
    # per image: detect person, point person, query count, caption (auto) + detect helmet,
    # query helmet (fixed; caption and person dedupe) = 6
    # + one probe per image: "forklift" (DEFAULT_PROBES; helmet is a fixed prompt, person listed)
    assert d["items"] == 28 and d["unlabelled"] == 0 and d["detect"] == 12 and d["query"] == 8
    # the stub student draws a box for every detect, so the 0-box probes disagree and wait for
    # a human; everything else agreed
    assert d["trainable"] == 24 and d["pending"] == 4
    assert client.post(f"/api/datasets/{ds['id']}/label").status_code == 409
    # a server restart loses the job: wipe answers + verdicts of half the items and the
    # in-memory job, then "接著跑" must redo exactly that and end with nothing unfinished
    from vlmforge.server.workspace import ALL

    for it in client.ws.items(q["id"], None, ALL)[:12]:
        client.ws.set_pred(it["id"], None, None)
        client.ws.db.execute(
            "update items set judge = null, verdict = null, status = 'pending' where id = ?",
            (it["id"],),
        )
    client.ws.db.commit()
    client.app.state.jobs.pop(q["id"], None)
    b = client.get(f"/api/queues/{q['id']}").json()
    assert b["job"] is None and b["unfinished"]["unpredicted"] == 12
    r = client.post(f"/api/queues/{q['id']}/resume").json()
    assert r["job"]["state"] == "running" and r["job"]["resumed"] is True
    r = wait(client, q["id"])
    assert r["job"]["state"] == "done"
    assert r["unfinished"] == {"images": 0, "unpredicted": 0, "unjudged": 0}
    assert client.post(f"/api/queues/{q['id']}/resume").status_code == 409
    d = client.get(f"/api/datasets/{ds['id']}").json()
    assert d["trainable"] == 24
    # rows per split and export
    train_rows = client.ws and __import__(
        "vlmforge.server.datasets", fromlist=["DatasetStore"]
    ).DatasetStore(client.ws).rows(ds["id"], "train")
    assert len(train_rows) + d["val_images"] * 6 == 24
    ex = client.post(f"/api/datasets/{ds['id']}/export").json()
    assert ex["rows"] == 24 and ex["images"] == 4
    assert client.get(f"/api/datasets/{ds['id']}/download/jsonl").status_code == 200
    # train on the dataset: pipeline gets train rows + a val manifest when the split is non-empty
    seen = {}

    def fake_run(self, name, st, cfg):
        seen["st"] = dict(st)
        st["state"], st["phase"] = "done", "done"
        self._write(name, st)

    monkeypatch.setattr(pl.Pipeline, "_run", fake_run)
    st = client.post(
        "/api/train", json={"name": "ft-ds", "steps": 40, "dataset_ids": [ds["id"]]}
    ).json()
    assert st["rows"] == len(train_rows) and st["datasets"] == ["工廠-1025"]
    assert st["region_head"] is False
    for _ in range(50):
        if client.get("/api/train/ft-ds").json()["state"] == "done":
            break
        time.sleep(0.02)
    cfg = __import__("yaml").safe_load(open(client.tmp / "runs" / "ft-ds" / "config.yaml"))
    assert cfg["region_head"] is False
    assert cfg["data"]["val_images"] == 0  # 4 images: nothing more may be held out
    d2 = client.get(f"/api/datasets/{ds['id']}").json()
    assert d2["trainable_images"] == 4 - d2["val_images"]  # val images never count
    # the Orin bundle needs an adapter file
    assert client.get("/api/train/ft-ds/orin.zip").status_code == 404
    (client.tmp / "runs" / "ft-ds" / "ft-ds.pt").write_bytes(b"x")
    r = client.get("/api/train/ft-ds/orin.zip")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert set(names) == set(BUNDLE_FILES)
    client.delete(f"/api/datasets/{ds['id']}")
    assert client.get("/api/datasets").json() == []


def test_write_config_always_validates_when_it_can(tmp_path, monkeypatch):
    """ft-1023-1911 trained blind: the dataset's val split had no confirmed question and two
    images were too few to split. The datasets' val rows win; otherwise ~10 % of ≥ 5 images."""
    import yaml

    from vlmforge import paths
    from vlmforge.server.pipeline import write_config

    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    data = lambda name, n, val=None: yaml.safe_load(  # noqa: E731
        write_config(name, tmp_path / "t.jsonl", n, 100, 2e-4, 16, False, val).read_text()
    )["data"]
    with_val = data("a", 113, [tmp_path / "val.jsonl"])
    assert with_val["val_images"] == 0 and with_val["val"] == [str(tmp_path / "val.jsonl")]
    assert data("b", 113)["val_images"] == 11
    assert data("c", 6)["val_images"] == 1
    assert data("d", 2)["val_images"] == 0


def test_remove_images_takes_their_questions_and_owned_files(client, tmp_path):
    """Removing an image drops it, its questions in this dataset and (for an upload nothing
    else uses) its file; a folder file stays on disk; 接著跑 no longer counts it."""
    ds = client.post("/api/datasets", json={"name": "rm"}).json()
    other = client.post("/api/datasets", json={"name": "other"}).json()
    up = [
        client.post(
            f"/api/datasets/{ds['id']}/upload", files={"file": (f"{c}.png", png(c), "image/png")}
        ).json()["image_ids"][0]
        for c in ("red", "green", "blue")
    ]
    folder = tmp_path / "cam1"
    folder.mkdir()
    (folder / "f.png").write_bytes(png("white"))
    fid = client.post(f"/api/datasets/{ds['id']}/folder", json={"dir": str(folder)}).json()[
        "image_ids"
    ][0]
    client.post(
        f"/api/datasets/{other['id']}/upload", files={"file": ("g.png", png("green"), "image/png")}
    )
    q = wait(client, client.post(f"/api/datasets/{ds['id']}/label").json()["id"])
    before = client.get(f"/api/datasets/{ds['id']}").json()
    per_image = before["items"] // 4
    red, green = up[0], up[1]
    red_path = client.ws.image(red)["path"]
    r = client.post(
        f"/api/datasets/{ds['id']}/images/remove", json={"image_ids": [red, green, fid]}
    ).json()
    assert r == {"removed": 3, "items": 3 * per_image, "files": 1}  # green is in "other"
    after = client.get(f"/api/datasets/{ds['id']}").json()
    assert after["images"] == 1 and after["items"] == per_image
    assert not Path(red_path).exists() and (folder / "f.png").exists()
    assert client.get(f"/api/datasets/{other['id']}").json()["images"] == 1
    assert client.get(f"/api/queues/{q['id']}").json()["unfinished"]["images"] == 0
    assert {c["image_id"] for c in client.get(f"/api/datasets/{ds['id']}/grid").json()} == {up[2]}


def test_relabel_keeps_human_answers_and_asks_the_rest_again(client):
    """Relabelling replaces machine questions with ones from the current settings; a question a
    human saved stays and is not asked twice; the old batch does not turn 「中斷」."""
    ds = client.post("/api/datasets", json={"name": "re", "config": {"probes": []}}).json()
    for c in ("red", "green"):
        client.post(
            f"/api/datasets/{ds['id']}/upload", files={"file": (f"{c}.png", png(c), "image/png")}
        )
    old = wait(client, client.post(f"/api/datasets/{ds['id']}/label").json()["id"])
    items = client.get(f"/api/datasets/{ds['id']}/grid").json()
    img = items[0]["image_id"]
    rev = client.get("/api/review/image/" + img).json()["items"]
    human = next(it for it in rev if it["skill"] == "query")
    client.post(
        f"/api/items/{human['id']}/review",
        json={"action": "correct", "reviewer": "P", "label": {"text": "7"}},
    )
    before = client.get(f"/api/datasets/{ds['id']}").json()["items"]
    # the settings change: a probe object is now asked on every image
    client.post(f"/api/datasets/{ds['id']}", json={"config": {"probes": ["forklift"]}})
    r = client.post(f"/api/datasets/{ds['id']}/relabel", json={"image_ids": [img]}).json()
    assert r["reset"] == {"dropped": len(rev) - 1, "kept": 1}
    wait(client, r["id"])
    after = client.get("/api/review/image/" + img).json()["items"]
    keys = [(it["skill"], it["prompt"]) for it in after]
    assert len(keys) == len(set(keys))  # nothing asked twice
    assert ("detect", "forklift") in keys  # the new setting applies
    kept = next(it for it in after if it["id"] == human["id"])
    assert kept["label"] == {"text": "7"} and kept["status"] == "corrected"
    assert client.get(f"/api/datasets/{ds['id']}").json()["items"] == before + 1
    assert client.get(f"/api/queues/{old['id']}").json()["unfinished"]["images"] == 0


def test_add_a_question_while_reviewing(client):
    ds = client.post("/api/datasets", json={"name": "add"}).json()
    img = client.post(
        f"/api/datasets/{ds['id']}/upload", files={"file": ("a.png", png("red"), "image/png")}
    ).json()["image_ids"][0]
    body = {"image_id": img, "skill": "detect", "prompt": " forklift ", "reviewer": "P"}
    it = client.post(f"/api/datasets/{ds['id']}/items", json={**body, "action": "pending"}).json()
    assert it["status"] == "pending" and it["prompt"] == "forklift" and it["label"] is None
    nxt = client.get("/api/review/next-image", params={"reviewer": "P", "dataset_id": ds["id"]})
    assert [i["id"] for i in nxt.json()["items"]] == [it["id"]]
    bad = client.post(f"/api/datasets/{ds['id']}/items", json={**body, "prompt": " "})
    assert bad.status_code == 422
    assert client.get(f"/api/datasets/{ds['id']}").json()["batches"] == []  # not a 「上一輪」
    # a new dataset with the same name must not inherit the deleted one's hand-made queue
    client.delete(f"/api/datasets/{ds['id']}")
    ds2 = client.post("/api/datasets", json={"name": "add"}).json()
    img2 = client.post(
        f"/api/datasets/{ds2['id']}/upload", files={"file": ("b.png", png("blue"), "image/png")}
    ).json()["image_ids"][0]
    client.post(
        f"/api/datasets/{ds2['id']}/items",
        json={**body, "image_id": img2, "action": "pending"},
    )
    assert client.get(f"/api/datasets/{ds2['id']}").json()["items"] == 1
