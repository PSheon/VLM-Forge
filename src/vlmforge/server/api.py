"""HTTP API of the supervision UI (GOALS §6). ``create_app(workspace, engine)``.

Workbench: ``POST /api/infer`` then ``POST /api/items`` (save a reviewed sample).
Queue: ``POST /api/queues`` (items + background predictions), ``GET /api/queues/{id}/next``
(claim), ``POST /api/items/{id}/review``. Compare: ``POST /api/compare`` (two models, one
image) and ``GET /api/heldout/diff`` (two eval reports). Runs: ``GET /api/runs``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__, paths
from .export import export_incoming, export_train
from .workspace import SKILLS, Workspace

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


class InferIn(BaseModel):
    image_id: str
    skill: str
    prompt: str = ""
    model: str = "base"


class CompareIn(BaseModel):
    image_id: str
    skill: str
    prompt: str = ""
    models: list[str]


class ItemIn(BaseModel):
    image_id: str
    skill: str
    prompt: str = ""
    pred: dict | None = None
    pred_model: str | None = None
    action: str = "correct"
    label: dict | None = None
    reviewer: str


class QueueIn(BaseModel):
    name: str
    skill: str
    prompts: list[str]
    model: str = "base"
    image_ids: list[str] | None = None
    source: str | None = None
    note: str = ""


class ReviewIn(BaseModel):
    action: str
    label: dict | None = None
    reviewer: str


class ScanIn(BaseModel):
    dir: str
    source: str


class ExportIn(BaseModel):
    name: str
    queue_ids: list[int] | None = None
    incoming: bool = False


def list_models() -> list[dict]:
    """``base`` plus every adapter under runs/ and adapters/ with its metadata."""
    out = [{"name": "base", "label": "base (Moondream 3.1)"}]
    for pt in sorted(list(paths.RUNS.glob("*/*.pt")) + list(paths.ADAPTERS.glob("**/*.pt"))):
        if pt.name == "resume.pt":
            continue
        meta = pt.with_suffix(".json")
        info = json.loads(meta.read_text()) if meta.exists() else {}
        out.append({"name": str(pt), "label": pt.stem, "meta": info})
    return out


def list_runs() -> list[dict]:
    runs = []
    if not paths.RUNS.exists():
        return runs
    for d in sorted(p for p in paths.RUNS.iterdir() if p.is_dir() and p.name != "baselines"):
        log = d / "log.jsonl"
        last, steps = None, []
        if log.exists():
            for line in open(log):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                steps.append(rec)
                last = rec
        metas = [
            json.loads(p.read_text())
            for p in d.glob("*.json")
            if p.name != "replay.json" and not p.name.startswith("eval-")
        ]
        evals = {}
        for p in d.glob("eval-*.json"):
            try:
                evals[p.name] = {
                    k: v
                    for k, v in json.loads(p.read_text())["metrics"].items()
                    if k.endswith("/macro")
                }
            except (KeyError, json.JSONDecodeError):
                continue
        runs.append(
            {
                "name": d.name,
                "adapters": [str(p) for p in d.glob("*.pt") if p.name != "resume.pt"],
                "meta": metas[0] if metas else None,
                "last": last,
                "curve": [
                    {"step": s["step"], "loss": s.get("loss"), "val": s.get("val/loss")}
                    for s in steps
                ],
                "evals": evals,
                "running": (d / "resume.pt").exists(),
            }
        )
    return runs


def list_reports() -> list[dict]:
    out = []
    for p in sorted(list(paths.BASELINES.glob("*.json")) + list(paths.RUNS.glob("*/eval-*.json"))):
        try:
            rep = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "path": str(p),
                "model": rep.get("model"),
                "manifest": rep.get("manifest"),
                "macro": {k: v for k, v in rep["metrics"].items() if k.endswith("/macro")},
            }
        )
    return out


def create_app(ws: Workspace, engine, web_dist: Path | None = None) -> FastAPI:
    app = FastAPI(title="VLM-Forge", version=__version__)
    jobs: dict[int, dict] = {}

    # ------------------------------------------------------------------ status
    @app.get("/api/status")
    def status():
        return {
            "version": __version__,
            "engine": engine.status(),
            "queues": ws.queues(),
            "jobs": jobs,
            "models": list_models(),
        }

    @app.get("/api/models")
    def models():
        return list_models()

    # ------------------------------------------------------------------ images
    @app.post("/api/images")
    async def upload(file: UploadFile):
        data = await file.read()
        ext = Path(file.filename or "x.jpg").suffix.lower().lstrip(".") or "jpg"
        return ws.add_image_bytes(data, "upload", ext)

    @app.post("/api/images/scan")
    def scan(body: ScanIn):
        d = Path(body.dir)
        if not d.is_dir():
            raise HTTPException(404, f"{d} is not a directory")
        added = [
            ws.add_image_path(p, body.source)["id"]
            for p in sorted(d.rglob("*"))
            if p.suffix.lower() in IMAGE_EXT
        ]
        return {"added": len(added), "image_ids": added}

    @app.get("/api/images")
    def images(source: str | None = None, limit: int = 500):
        return ws.images(source, limit)

    @app.get("/api/images/{image_id}")
    def image(image_id: str):
        im = ws.image(image_id)
        if im is None:
            raise HTTPException(404)
        return im

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: str):
        im = ws.image(image_id)
        if im is None or not Path(im["path"]).exists():
            raise HTTPException(404)
        return FileResponse(im["path"])

    # ------------------------------------------------------------------ inference
    def _infer(image_id: str, skill: str, prompt: str, model: str) -> dict:
        im = ws.image(image_id)
        if im is None:
            raise HTTPException(404, "no such image")
        if skill not in SKILLS:
            raise HTTPException(422, f"unknown skill {skill}")
        t0 = time.time()
        pred = engine.infer([im["path"]], skill, prompt, model)[0]
        return {"pred": pred, "model": model, "ms": round((time.time() - t0) * 1000)}

    @app.post("/api/infer")
    def infer(body: InferIn):
        return _infer(body.image_id, body.skill, body.prompt, body.model)

    @app.post("/api/compare")
    def compare(body: CompareIn):
        return {m: _infer(body.image_id, body.skill, body.prompt, m) for m in body.models}

    # ------------------------------------------------------------------ workbench save
    @app.post("/api/items")
    def save_item(body: ItemIn):
        item_id = ws.add_item(
            body.image_id, body.skill, body.prompt, body.pred, body.pred_model, None, "workbench"
        )
        try:
            return ws.review(item_id, body.action, body.reviewer, body.label)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    # ------------------------------------------------------------------ queues
    def _predict_queue(qid: int, model: str):
        job = jobs[qid]
        try:
            items = ws.items(qid, "pending")
            by_prompt: dict[tuple[str, str], list[dict]] = {}
            for it in items:
                if it["pred"] is None:
                    by_prompt.setdefault((it["skill"], it["prompt"]), []).append(it)
            job["total"] = sum(len(v) for v in by_prompt.values())
            for (skill, prompt), its in by_prompt.items():
                for i in range(0, len(its), 32):
                    chunk = its[i : i + 32]
                    preds = engine.infer([c["path"] for c in chunk], skill, prompt, model)
                    for c, p in zip(chunk, preds, strict=True):
                        ws.set_pred(c["id"], p, model)
                    job["done"] += len(chunk)
            job["state"] = "done"
        except Exception as e:  # noqa: BLE001 - shown in the UI
            job["state"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"

    @app.post("/api/queues")
    def create_queue(body: QueueIn):
        if body.skill not in SKILLS:
            raise HTTPException(422, f"unknown skill {body.skill}")
        ids = body.image_ids
        if ids is None:
            if not body.source:
                raise HTTPException(422, "image_ids or source required")
            ids = [im["id"] for im in ws.images(body.source, 100000)]
        if not ids:
            raise HTTPException(422, "no images")
        try:
            q = ws.create_queue(body.name, body.skill, body.model, body.note)
        except Exception as e:  # noqa: BLE001 - duplicate name
            raise HTTPException(409, f"queue exists? {e}") from e
        for image_id in ids:
            for prompt in body.prompts:
                ws.add_item(image_id, body.skill, prompt, None, None, q["id"])
        jobs[q["id"]] = {"state": "running", "done": 0, "total": 0, "error": None}
        threading.Thread(target=_predict_queue, args=(q["id"], body.model), daemon=True).start()
        return ws.queue(q["id"])

    @app.get("/api/queues")
    def queues():
        return [{**q, "job": jobs.get(q["id"])} for q in ws.queues()]

    @app.get("/api/queues/{qid}")
    def queue(qid: int):
        try:
            return {**ws.queue(qid), "job": jobs.get(qid)}
        except KeyError as e:
            raise HTTPException(404) from e

    @app.get("/api/queues/{qid}/items")
    def queue_items(qid: int, status: str | None = None):
        return ws.items(qid, status)

    @app.get("/api/queues/{qid}/next")
    def queue_next(qid: int, reviewer: str, after: int | None = None):
        it = ws.claim_next(qid, reviewer, after)
        return it or {}

    @app.get("/api/items/{item_id}")
    def item(item_id: int):
        it = ws.item(item_id)
        if it is None:
            raise HTTPException(404)
        return it

    @app.post("/api/items/{item_id}/review")
    def review(item_id: int, body: ReviewIn):
        try:
            return ws.review(item_id, body.action, body.reviewer, body.label)
        except KeyError as e:
            raise HTTPException(404) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/api/items/{item_id}/reopen")
    def reopen(item_id: int):
        return ws.reopen(item_id)

    # ------------------------------------------------------------------ held-out compare
    @app.get("/api/heldout/reports")
    def reports():
        return list_reports()

    @app.get("/api/heldout/diff")
    def heldout_diff(base: str, adapter: str, only_changed: bool = True):
        """Item pairs of two eval reports on the same manifest; images registered on demand so
        the canvas can show them."""
        a, b = json.loads(Path(base).read_text()), json.loads(Path(adapter).read_text())
        if a.get("manifest") != b.get("manifest"):
            raise HTTPException(422, "reports are on different manifests")
        rows = [json.loads(line) for line in open(a["manifest"])]
        out = []
        for r, ia, ib in zip(rows, a["items"], b["items"], strict=True):
            same = ia.get("tp_fp_fn") == ib.get("tp_fp_fn") and ia.get("ok") == ib.get("ok")
            if only_changed and same:
                continue
            img = ws.add_image_path(Path(r["image"]), "heldout")
            out.append({"row": r, "image_id": img["id"], "base": ia, "adapter": ib})
        return {"manifest": a["manifest"], "n": len(rows), "items": out}

    # ------------------------------------------------------------------ runs / export
    @app.get("/api/runs")
    def runs():
        return list_runs()

    @app.post("/api/export")
    def export(body: ExportIn):
        res = {"train": export_train(ws, body.queue_ids, body.name)}
        if body.incoming:
            res["incoming"] = export_incoming(
                ws, body.queue_ids, body.name, paths.WORKSPACE / "incoming"
            )
        return res

    if web_dist and web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app
