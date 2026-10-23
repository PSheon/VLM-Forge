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
    judge: bool = True


class AutoQueueIn(BaseModel):
    """Autogen (GOALS §6): the tool picks ``n_images`` from ``sources``, the teacher writes the
    questions, the student answers, agreement + judge decide, humans check per image."""

    name: str
    sources: list[str]
    n_images: int = 50
    skills: list[str] = ["detect", "point", "query", "caption"]
    model: str = "base"
    focus: str = ""
    topics: list[str] = []
    seed: int = 0


class ImageReviewIn(BaseModel):
    reviewer: str
    items: list[dict]  # {"id", "action": approve|correct|discard, "label"?}


class JudgeIn(BaseModel):
    image_id: str
    skill: str
    prompt: str = ""
    pred: dict


class ReviewIn(BaseModel):
    action: str
    label: dict | None = None
    reviewer: str


class ScanIn(BaseModel):
    dir: str
    source: str
    limit: int | None = None
    seed: int = 0


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
    for d in sorted(p for p in paths.RUNS.iterdir() if p.is_dir()):
        log = d / "log.jsonl"
        if not (log.exists() or any(d.glob("*.pt")) or any(d.glob("*.json"))):
            continue  # not a run (runs/logs, runs/baselines)
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
        """Register images under a directory; ``limit`` takes a seeded random sample so a
        Data Core release with 800k files can feed the autogen pool without indexing it all."""
        import random

        d = Path(body.dir)
        if not d.is_dir():
            raise HTTPException(404, f"{d} is not a directory")
        files = [p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXT]
        files.sort()
        if body.limit and len(files) > body.limit:
            files = random.Random(body.seed).sample(files, body.limit)
        added = [ws.add_image_path(p, body.source)["id"] for p in files]
        return {"added": len(added), "found": len(files), "image_ids": added}

    @app.get("/api/images")
    def images(source: str | None = None, limit: int = 500):
        return ws.images(source, limit)

    @app.get("/api/sources")
    def sources():
        return ws.sources()

    @app.get("/api/images/{image_id}")
    def image(image_id: str):
        im = ws.image(image_id)
        if im is None:
            raise HTTPException(404)
        return im

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: str, w: int | None = None):
        """The image, or a cached JPEG resized to width ``w`` (thumbnails, display copies of
        the 2560 px CCTV frames) under workspace/cache/."""
        im = ws.image(image_id)
        if im is None or not Path(im["path"]).exists():
            raise HTTPException(404)
        if not w or w >= im["width"]:
            return FileResponse(im["path"])
        w = max(32, min(w, 4096))
        cache = ws.root / "cache" / f"{image_id}-w{w}.jpg"
        if not cache.exists():
            from PIL import Image

            cache.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(im["path"]).convert("RGB")
            img.thumbnail((w, w * 4), Image.LANCZOS)
            img.save(cache, "JPEG", quality=85)
        return FileResponse(cache, headers={"Cache-Control": "max-age=86400"})

    # ------------------------------------------------------------------ inference
    def _infer(image_id: str, skill: str, prompt: str, model: str) -> dict:
        im = ws.image(image_id)
        if im is None:
            raise HTTPException(404, "no such image")
        if skill not in SKILLS:
            raise HTTPException(422, f"unknown skill {skill}")
        t0 = time.time()
        try:
            pred = engine.infer([im["path"]], skill, prompt, model)[0]
        except Exception as e:  # noqa: BLE001 - the worker's error, shown in the UI
            raise HTTPException(503, f"{type(e).__name__}: {e}") from e
        return {"pred": pred, "model": model, "ms": round((time.time() - t0) * 1000)}

    @app.post("/api/infer")
    def infer(body: InferIn):
        return _infer(body.image_id, body.skill, body.prompt, body.model)

    @app.post("/api/compare")
    def compare(body: CompareIn):
        return {m: _infer(body.image_id, body.skill, body.prompt, m) for m in body.models}

    @app.post("/api/judge")
    def judge_one(body: JudgeIn):
        im = ws.image(body.image_id)
        if im is None:
            raise HTTPException(404, "no such image")
        t0 = time.time()
        try:
            v = engine.judge_items(
                [
                    {
                        "path": im["path"],
                        "skill": body.skill,
                        "prompt": body.prompt,
                        "pred": body.pred,
                    }
                ]
            )[0]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"{type(e).__name__}: {e}") from e
        return {**v, "ms": round((time.time() - t0) * 1000)}

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
    def _judge_queue(qid: int, job: dict):
        """Judge every predicted, unjudged pending item, then triage (GOALS §6)."""
        job["phase"] = "judging"
        todo = ws.unjudged(qid)
        job["total"], job["done"] = len(todo), 0
        for i in range(0, len(todo), 8):
            chunk = todo[i : i + 8]
            for it, v in zip(chunk, engine.judge_items(chunk), strict=True):
                ws.set_judge(it["id"], v)
            job["done"] += len(chunk)
        job["triage"] = ws.triage(qid)

    def _predict_queue(qid: int, model: str, judge: bool):
        job = jobs[qid]
        try:
            items = ws.items(qid, "pending")
            by_prompt: dict[tuple[str, str], list[dict]] = {}
            for it in items:
                if it["pred"] is None:
                    by_prompt.setdefault((it["skill"], it["prompt"]), []).append(it)
            job["phase"] = "predicting"
            job["total"] = sum(len(v) for v in by_prompt.values())
            for (skill, prompt), its in by_prompt.items():
                for i in range(0, len(its), 32):
                    chunk = its[i : i + 32]
                    preds = engine.infer([c["path"] for c in chunk], skill, prompt, model)
                    for c, p in zip(chunk, preds, strict=True):
                        ws.set_pred(c["id"], p, model)
                    job["done"] += len(chunk)
            if judge:
                _judge_queue(qid, job)
            job["state"] = "done"
        except Exception as e:  # noqa: BLE001 - shown in the UI
            job["state"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"

    def _start(qid: int, target, *args):
        jobs[qid] = {"state": "running", "phase": "", "done": 0, "total": 0, "error": None}
        threading.Thread(target=target, args=(qid, *args), daemon=True).start()

    @app.post("/api/queues/{qid}/judge")
    def judge_queue(qid: int):
        """(Re)run the judge on this queue's unjudged pending items."""
        if jobs.get(qid, {}).get("state") == "running":
            raise HTTPException(409, "job running")

        def run(q):
            job = jobs[q]
            try:
                _judge_queue(q, job)
                job["state"] = "done"
            except Exception as e:  # noqa: BLE001
                job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"

        _start(qid, run)
        return {**ws.queue(qid), "job": jobs[qid]}

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
        judge = Path(paths.JUDGE_MODEL).name if body.judge else ""
        try:
            q = ws.create_queue(body.name, body.skill, body.model, body.note, judge)
        except Exception as e:  # noqa: BLE001 - duplicate name
            raise HTTPException(409, f"queue exists? {e}") from e
        for image_id in ids:
            for prompt in body.prompts:
                ws.add_item(image_id, body.skill, prompt, None, None, q["id"])
        _start(q["id"], _predict_queue, body.model, body.judge)
        return {**ws.queue(q["id"]), "job": jobs[q["id"]]}

    def _autogen_queue(qid: int, cfg: dict):
        """generating (teacher) -> predicting (student) -> judging (agreement; judge for
        captions) -> triage by image."""
        from ..autogen import agree, items_from_teacher

        job = jobs[qid]
        try:
            skills = set(cfg["skills"])
            images = cfg["image_ids"]
            job["phase"], job["total"], job["done"] = "generating", len(images), 0
            for image_id in images:
                im = ws.image(image_id)
                t = engine.teacher_generate(im["path"], cfg.get("focus", ""), cfg.get("topics"))
                for it in items_from_teacher(t, skills):
                    ws.add_item(
                        image_id, it["skill"], it["prompt"], None, None, qid, "auto", it["teacher"]
                    )
                job["done"] += 1
            # student
            items = ws.items(qid, "pending")
            by_prompt: dict[tuple[str, str], list[dict]] = {}
            for it in items:
                by_prompt.setdefault((it["skill"], it["prompt"]), []).append(it)
            job["phase"], job["total"], job["done"] = "predicting", len(items), 0
            try:
                for (skill, prompt), its in by_prompt.items():
                    for i in range(0, len(its), 32):
                        chunk = its[i : i + 32]
                        preds = engine.infer(
                            [c["path"] for c in chunk], skill, prompt, cfg["model"]
                        )
                        for c, p in zip(chunk, preds, strict=True):
                            ws.set_pred(c["id"], p, cfg["model"])
                        job["done"] += len(chunk)
            except Exception as e:  # noqa: BLE001 - no student (weights missing): humans
                job["student_error"] = f"{type(e).__name__}: {e}"  # review the teacher alone
            # agreement, and the judge for the teacher's captions
            items = ws.items(qid, "pending")
            caps = [it for it in items if it["skill"] == "caption"]
            job["phase"], job["total"], job["done"] = "judging", len(items), 0
            for it in items:
                if it["skill"] != "caption":
                    ws.set_judge(it["id"], agree(it["skill"], it["pred"], it["teacher"]))
                    job["done"] += 1
            for i in range(0, len(caps), 8):
                chunk = caps[i : i + 8]
                asks = [{**c, "pred": c["teacher"]} for c in chunk]  # judge the teacher caption
                for c, v in zip(chunk, engine.judge_items(asks), strict=True):
                    ws.set_judge(c["id"], v)
                job["done"] += len(chunk)
            job["triage"] = ws.triage_images(qid, cfg.get("seed", 0))
            job["state"] = "done"
        except Exception as e:  # noqa: BLE001 - shown in the UI
            job["state"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"

    @app.post("/api/queues/auto")
    def create_auto_queue(body: AutoQueueIn):
        import random

        bad = [s for s in body.skills if s not in SKILLS]
        if bad or not body.skills:
            raise HTTPException(422, f"bad skills {bad}")
        pool: list[str] = []
        for s in body.sources:
            if s == "heldout" or s.startswith("heldout"):
                raise HTTPException(422, "held-out images never become training data (GOALS §4)")
            pool += [im["id"] for im in ws.images(s, 100000)]
        pool = sorted(set(pool))
        if not pool:
            raise HTTPException(422, "no images in those sources")
        rng = random.Random(body.seed)
        rng.shuffle(pool)
        chosen = pool[: body.n_images]
        judge = Path(paths.JUDGE_MODEL).name
        try:
            q = ws.create_queue(body.name, "query", body.model, "autogen", judge)
        except Exception as e:  # noqa: BLE001 - duplicate name
            raise HTTPException(409, f"queue exists? {e}") from e
        cfg = {**body.model_dump(), "image_ids": chosen}
        ws.set_queue_mode(q["id"], "auto", cfg)
        _start(q["id"], _autogen_queue, cfg)
        return {**ws.queue(q["id"]), "job": jobs[q["id"]]}

    @app.get("/api/queues/{qid}/images")
    def queue_images(qid: int):
        return ws.image_summary(qid)

    @app.get("/api/queues/{qid}/next-image")
    def queue_next_image(qid: int, reviewer: str, after: str | None = None):
        r = ws.claim_next_image(qid, reviewer, after)
        if r is None:
            return {}
        image, items = r
        return {"image": image, "items": items}

    @app.get("/api/queues/{qid}/image/{image_id}")
    def queue_image(qid: int, image_id: str):
        im = ws.image(image_id)
        if im is None:
            raise HTTPException(404)
        return {"image": im, "items": [it for it in ws.items(qid) if it["image_id"] == image_id]}

    @app.post("/api/queues/{qid}/review-image")
    def review_image(qid: int, body: ImageReviewIn):
        out = []
        for d in body.items:
            try:
                out.append(
                    ws.review(
                        int(d["id"]), d.get("action", "approve"), body.reviewer, d.get("label")
                    )
                )
            except (KeyError, ValueError) as e:
                raise HTTPException(422, f"item {d.get('id')}: {e}") from e
        return out

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
