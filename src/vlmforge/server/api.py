"""HTTP API of the supervision UI (GOALS §6). ``create_app(workspace, engine)``.

Workbench: ``POST /api/infer`` then ``POST /api/items`` (save a reviewed sample).
Queue: ``POST /api/queues`` (items + background predictions), ``GET /api/queues/{id}/next``
(claim), ``POST /api/items/{id}/review``. Compare: ``POST /api/compare`` (two models, one
image) and ``GET /api/heldout/diff`` (two eval reports). Runs: ``GET /api/runs``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__, paths
from ..orin import make_bundle
from .datasets import DatasetStore
from .export import export_incoming, export_train
from .pipeline import DEFAULT_LR, DEFAULT_STEPS, Pipeline, default_name
from .workspace import ALL, IMAGE_EXT, SKILLS, Workspace

GEN_BATCH = 8  # images per teacher call (GPU throughput; the 32B judge is the slow part)


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


class DataScanIn(BaseModel):
    dir: str
    limit: int | None = None


class DataAutogenIn(BaseModel):
    source: str
    n_images: int | None = None  # default: every image of the source
    skills: list[str] = ["detect", "point", "query", "caption"]
    topics: list[str] = []
    focus: str = ""
    model: str = "base"


class TrainIn(BaseModel):
    name: str | None = None
    steps: int = DEFAULT_STEPS
    lr: float = DEFAULT_LR
    rank: int = 16
    dataset_ids: list[int] = []  # empty = everything reviewed in the workspace
    region_head: bool = False  # True breaks Photon loading (GOALS §3)


class DatasetIn(BaseModel):
    name: str
    config: dict = {}
    note: str = ""


class DatasetUpdateIn(BaseModel):
    name: str | None = None
    config: dict | None = None


class FolderIn(BaseModel):
    dir: str
    limit: int | None = None


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


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _with_backoff(fn, paths_: list[str]):
    """Run a batched GPU call; on CUDA OOM (the GB300 is shared, GOALS §3) halve the batch
    down to one image before giving up."""
    try:
        return fn(paths_)
    except Exception as e:  # noqa: BLE001
        if not _is_oom(e) or len(paths_) == 1:
            raise
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        mid = len(paths_) // 2
        return _with_backoff(fn, paths_[:mid]) + _with_backoff(fn, paths_[mid:])


def create_app(ws: Workspace, engine, web_dist: Path | None = None) -> FastAPI:
    app = FastAPI(title="VLM-Forge", version=__version__)
    jobs: dict[int, dict] = {}  # in memory: a restart loses them (hence /resume)
    app.state.jobs = jobs

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

    @app.post("/api/status/clear-error")
    def clear_error():
        engine.clear_error()
        return {"ok": True}

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
            # write-then-rename: a second request for the same image must never be served
            # a half-written file (that showed up as images that never finished loading)
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            img.save(tmp, "JPEG", quality=85)
            os.replace(tmp, cache)
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
            items = ws.items(qid, "pending", ALL)
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

    def _batch(q: dict) -> dict:
        """A queue row for the UI: its in-memory job, and for autogen batches what is still
        unfinished (so an interrupted batch shows 「接著跑」 instead of looking done)."""
        out = {**q, "job": jobs.get(q["id"])}
        if q.get("mode") == "auto" and q.get("config"):
            try:
                out["unfinished"] = ws.unfinished(q["id"], json.loads(q["config"])["image_ids"])
            except (KeyError, ValueError, TypeError):
                pass
        out.pop("config", None)
        return out

    def _start(qid: int, target, *args):
        jobs[qid] = {"state": "running", "phase": "", "done": 0, "total": 0, "error": None}
        threading.Thread(target=target, args=(qid, *args), daemon=True).start()

    @app.post("/api/queues/{qid}/cancel")
    def cancel_queue(qid: int):
        """Stop a running batch after the current chunk; what was generated stays."""
        job = jobs.get(qid)
        if not job or job["state"] != "running":
            raise HTTPException(409, "not running")
        job["cancel"] = True
        return job

    @app.post("/api/queues/{qid}/resume")
    def resume_queue(qid: int):
        """Pick an autogen batch up where it stopped (server restart, cancel, error):
        unasked images, unanswered items, uncompared items, then triage."""
        try:
            q = ws.queue(qid)
        except KeyError as e:
            raise HTTPException(404) from e
        if q.get("mode") != "auto" or not q.get("config"):
            raise HTTPException(422, "只有自動標註的批次能接著跑")
        if jobs.get(qid, {}).get("state") == "running":
            raise HTTPException(409, "還在跑")
        cfg = json.loads(q["config"])
        if not any(ws.unfinished(qid, cfg["image_ids"]).values()):
            raise HTTPException(409, "這批已經完成，沒有要接的")
        _start(qid, _autogen_queue, cfg, True)
        jobs[qid]["resumed"] = True  # before the thread gets there, so the reply says so
        return _batch(q)

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

    def _autogen_queue(qid: int, cfg: dict, resume: bool = False):
        """generating (teacher) -> predicting (student) -> judging (agreement; judge for
        captions) -> triage by image. Every phase only does what is still missing, so the
        same function resumes a batch that a restart or a cancel interrupted."""
        from ..autogen import agree, items_from_teacher

        job = jobs[qid]
        try:
            from ..autogen import items_from_fixed, parse_fixed

            skills = set(cfg["skills"])
            images = cfg["image_ids"]
            if resume:
                have = {it["image_id"] for it in ws.items(qid, None, ALL)}
                images = [i for i in images if i not in have]
                job["resumed"] = True
            mode = cfg.get("mode", "auto")  # auto | fixed | both
            fixed = parse_fixed(cfg.get("fixed", "")) if mode in ("fixed", "both") else None
            job["phase"], job["total"], job["done"] = "generating", len(images), 0
            job["started"] = time.time()
            for k in range(0, len(images), GEN_BATCH):
                if job.get("cancel"):
                    job["state"] = "cancelled"
                    return
                chunk = images[k : k + GEN_BATCH]
                paths_ = [ws.image(i)["path"] for i in chunk]
                per_image: list[list[dict]] = [[] for _ in chunk]
                if mode in ("auto", "both"):
                    ts = _with_backoff(
                        lambda ps: engine.teacher_generate_batch(
                            ps, cfg.get("focus", ""), cfg.get("topics")
                        ),
                        paths_,
                    )
                    for j, t in enumerate(ts):
                        per_image[j] += items_from_teacher(t, skills)
                if fixed and any(fixed.values()):
                    ts = _with_backoff(lambda ps: engine.teacher_fixed_batch(ps, fixed), paths_)
                    for j, t in enumerate(ts):
                        per_image[j] += items_from_fixed(t, fixed)
                for image_id, its in zip(chunk, per_image, strict=True):
                    seen: set[tuple[str, str]] = set()
                    for it in its:
                        key = (it["skill"], it["prompt"])
                        if key in seen:
                            continue
                        seen.add(key)
                        ws.add_item(
                            image_id,
                            it["skill"],
                            it["prompt"],
                            None,
                            None,
                            qid,
                            "auto",
                            it["teacher"],
                        )
                job["done"] += len(chunk)
            if job.get("cancel"):
                job["state"] = "cancelled"
                return
            # student (only items without an answer: a resumed batch keeps what it had)
            items = [it for it in ws.items(qid, "pending", ALL) if it["pred"] is None]
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
            # agreement, and the judge for the teacher's captions (unjudged items only)
            items = [it for it in ws.items(qid, "pending", ALL) if not it["verdict"]]
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

    # ------------------------------------------------------------------ the four steps
    pipeline = Pipeline(ws)

    @app.post("/api/data/scan")
    def data_scan(body: DataScanIn):
        """① a folder on this machine becomes a data source named after it."""
        d = Path(body.dir).expanduser()
        if not d.is_dir():
            raise HTTPException(404, f"{d} 不是目錄")
        name = d.name
        if name.startswith("heldout"):
            raise HTTPException(422, "held-out 資料夾不能當訓練來源")
        r = scan(ScanIn(dir=str(d), source=name, limit=body.limit))
        return {"source": name, **r}

    @app.post("/api/data/autogen")
    def data_autogen(body: DataAutogenIn):
        """① run the whole generate -> answer -> compare chain on a source."""
        n = body.n_images or len(ws.images(body.source, 100000))
        qname = f"{body.source}-{time.strftime('%m%d-%H%M')}"
        return create_auto_queue(
            AutoQueueIn(
                name=qname,
                sources=[body.source],
                n_images=n,
                skills=body.skills,
                topics=body.topics,
                focus=body.focus,
                model=body.model,
            )
        )

    @app.get("/api/review/summary")
    def review_summary():
        return {
            **ws.review_summary(),
            "batches": [_batch(q) for q in ws.queues()],
            "sources": ws.sources(),
        }

    @app.get("/api/review/next-image")
    def review_next_image(reviewer: str, after: str | None = None, dataset_id: int | None = None):
        r = ws.claim_next_image(None, reviewer, after, dataset_id)
        if r is None:
            return {}
        image, items = r
        return {"image": image, "items": items}

    @app.get("/api/review/images")
    def review_images(status: str = "pending", dataset_id: int | None = None):
        """Images with items in ``status`` (pending | auto | reviewed | all), newest batch
        first; ``dataset_id`` narrows to one dataset. ``claimed_by`` names who holds it."""
        rows = ws.db.execute(
            "select items.image_id, count(*) n, sum(items.status = 'pending') pending,"
            " sum(verdict = 'fix' and items.status = 'pending') fix, max(spot) spot,"
            " sum(items.status = 'auto') auto, max(items.queue_id) queue_id,"
            " max(case when claimed_at >= ? then claimed_by end) claimed_by"
            " from items join queues on queues.id = items.queue_id"
            " where (? is null or queues.dataset_id = ?)"
            " group by items.image_id order by max(items.queue_id) desc, min(items.id)",
            (time.time() - 600, dataset_id, dataset_id),
        ).fetchall()
        out = [dict(r) for r in rows]
        if status == "pending":
            out = [r for r in out if r["pending"]]
        elif status == "auto":
            out = [r for r in out if r["auto"] == r["n"]]
        elif status == "reviewed":
            out = [r for r in out if not r["pending"] and r["auto"] != r["n"]]
        return out

    @app.get("/api/review/image/{image_id}")
    def review_image_get(image_id: str):
        im = ws.image(image_id)
        if im is None:
            raise HTTPException(404)
        return {"image": im, "items": ws.items_for_image(image_id)}

    @app.post("/api/review/image")
    def review_image_post(body: ImageReviewIn):
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

    # ------------------------------------------------------------------ datasets (GOALS §4)
    store = DatasetStore(ws)

    @app.get("/api/datasets")
    def datasets():
        return store.list()

    @app.post("/api/datasets")
    def dataset_create(body: DatasetIn):
        try:
            return store.create(body.name, body.config, body.note)
        except Exception as e:  # noqa: BLE001 - duplicate / blank name
            raise HTTPException(409, f"建不了：{e}") from e

    @app.get("/api/datasets/{ds_id}")
    def dataset_get(ds_id: int):
        try:
            d = store.get(ds_id)
        except KeyError as e:
            raise HTTPException(404) from e
        d["batches"] = [_batch(q) for q in ws.queues() if q.get("dataset_id") == ds_id]
        d["unlabelled"] = len(store.unlabelled_image_ids(ds_id))
        return d

    @app.post("/api/datasets/{ds_id}")
    def dataset_update(ds_id: int, body: DatasetUpdateIn):
        return store.update(ds_id, body.config, body.name)

    @app.delete("/api/datasets/{ds_id}")
    def dataset_delete(ds_id: int):
        store.delete(ds_id)
        return {"ok": True}

    @app.get("/api/datasets/{ds_id}/images")
    def dataset_images(ds_id: int, split: str | None = None):
        return [ws.image(i) for i in store.image_ids(ds_id, split)]

    @app.post("/api/datasets/{ds_id}/upload")
    async def dataset_upload(ds_id: int, file: UploadFile):
        data = await file.read()
        try:
            ids = store.add_bytes(ds_id, data, file.filename or "upload.jpg")
        except KeyError as e:
            raise HTTPException(404, "沒有這個資料集") from e
        return {"added": len(ids), "image_ids": ids}

    @app.post("/api/datasets/{ds_id}/folder")
    def dataset_folder(ds_id: int, body: FolderIn):
        d = Path(body.dir).expanduser()
        if not d.is_dir():
            raise HTTPException(404, f"{d} 不是目錄")
        if "heldout" in str(d):
            raise HTTPException(422, "held-out 的圖不能進資料集（GOALS §4）")
        try:
            ids = store.add_folder(ds_id, d, body.limit)
        except KeyError as e:
            raise HTTPException(404, "沒有這個資料集") from e
        return {"added": len(ids), "image_ids": ids}

    @app.post("/api/datasets/{ds_id}/label")
    def dataset_label(ds_id: int, n_images: int | None = None, model: str = "base"):
        """Run the teacher -> student -> compare chain on the dataset's unlabelled images,
        following its config (auto questions and/or fixed prompts)."""
        ds = store.get(ds_id)
        ids = store.unlabelled_image_ids(ds_id)
        if n_images:
            ids = ids[:n_images]
        if not ids:
            raise HTTPException(409, "沒有還沒標的圖")
        cfg = ds["config"]
        qname = f"{ds['name']}-{time.strftime('%m%d-%H%M%S')}"
        q = ws.create_queue(qname, "query", model, "autogen", Path(paths.JUDGE_MODEL).name)
        store.attach_queue(q["id"], ds_id)
        job_cfg = {
            "image_ids": ids,
            "skills": cfg.get("skills") or ["detect", "point", "query", "caption"],
            "mode": cfg.get("mode", "auto"),
            "fixed": cfg.get("fixed", ""),
            "topics": cfg.get("topics") or [],
            "focus": cfg.get("focus", ""),
            "model": model,
            "seed": 0,
        }
        ws.set_queue_mode(q["id"], "auto", job_cfg)
        _start(q["id"], _autogen_queue, job_cfg)
        return {**ws.queue(q["id"]), "job": jobs[q["id"]]}

    @app.post("/api/datasets/{ds_id}/export")
    def dataset_export(ds_id: int, images: bool = True):
        return store.export(ds_id, paths.WORKSPACE / "export", images)

    @app.get("/api/datasets/{ds_id}/download/{kind}")
    def dataset_download(ds_id: int, kind: str):
        ds = store.get(ds_id)
        fn = f"{ds['name']}.jsonl" if kind == "jsonl" else f"{ds['name']}-images.zip"
        p = paths.WORKSPACE / "export" / fn
        if not p.exists():
            raise HTTPException(404, "先按匯出")
        return FileResponse(p, filename=p.name)

    @app.post("/api/train")
    def train_start(body: TrainIn):
        """③ train on the chosen datasets' train splits (or everything reviewed), then
        evaluate on the held-out suites and each dataset's val split, then gate."""
        from ..data import write_rows

        name = body.name or default_name()
        rows: list[dict] | None = None
        vals: dict[str, Path] = {}
        names: list[str] = []
        if body.dataset_ids:
            rows = []
            for ds_id in body.dataset_ids:
                ds = store.get(ds_id)
                names.append(ds["name"])
                rows += store.rows(ds_id, "train")
                val_rows = store.rows(ds_id, "val")
                if val_rows:
                    vp = paths.WORKSPACE / "export" / f"val-{ds['name']}.jsonl"
                    write_rows(vp, val_rows, ensure_ascii=False)
                    vals[f"val-{ds['name']}"] = vp
        try:
            return pipeline.start(
                name, body.steps, body.lr, body.rank, rows, vals, body.region_head, names
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/api/train/{name}/cancel")
    def train_cancel(name: str):
        if not pipeline.cancel(name):
            raise HTTPException(409, "沒有在跑")
        return {"ok": True}

    @app.delete("/api/train/{name}")
    def train_delete(name: str):
        try:
            pipeline.delete(name)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"ok": True}

    @app.get("/api/datasets/{ds_id}/grid")
    def dataset_grid(ds_id: int):
        """Every image of the dataset with its split and review counts (the thumbnail grid)."""
        rows = ws.db.execute(
            "select di.image_id, di.split, count(items.id) n,"
            " sum(items.status = 'pending') pending,"
            " sum(items.verdict = 'fix' and items.status = 'pending') fix,"
            " sum(items.status = 'auto') auto"
            " from dataset_images di"
            " left join queues on queues.dataset_id = di.dataset_id"
            " left join items on items.queue_id = queues.id and items.image_id = di.image_id"
            " where di.dataset_id = ? group by di.image_id, di.split order by di.added_at",
            (ds_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    @app.post("/api/datasets/{ds_id}/items")
    def dataset_item(ds_id: int, body: ItemIn):
        """A confirmed item saved by hand (the playground's 存成資料集的題)."""
        ds = store.get(ds_id)
        qname = f"{ds['name']}-manual"
        try:
            q = ws.queue_by_name(qname)
        except KeyError:
            q = ws.create_queue(qname, "query", "base", "manual")
            store.attach_queue(q["id"], ds_id)
        store.add_image(ds_id, body.image_id)
        item_id = ws.add_item(
            body.image_id, body.skill, body.prompt, body.pred, body.pred_model, q["id"], "workbench"
        )
        try:
            return ws.review(item_id, body.action, body.reviewer, body.label)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/train/{name}/orin.zip")
    def train_orin(name: str):
        st = pipeline.status(name)
        if st is None:
            raise HTTPException(404)
        try:
            z = make_bundle(paths.RUNS / name, name, st)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        return FileResponse(z, filename=z.name)

    @app.get("/api/train")
    def train_list():
        return {"runs": pipeline.list(), "trainable": ws.review_summary()["trainable"]}

    @app.get("/api/train/{name}")
    def train_status(name: str):
        st = pipeline.status(name)
        if st is None:
            raise HTTPException(404)
        return st

    @app.get("/api/test/summary")
    def test_summary():
        """④ base vs every adapter on the held-out suites, plus the gate verdict."""
        base = {}
        for s in paths.HELDOUT_SUITES:
            p = paths.BASELINES / f"base-{s}.json"
            if p.exists():
                m = json.loads(p.read_text())["metrics"]
                base[s] = {k.split("/")[0]: v for k, v in m.items() if k.endswith("/macro")}
        return {"suites": list(paths.HELDOUT_SUITES), "base": base, "runs": pipeline.list()}

    @app.get("/api/queues")
    def queues():
        return [_batch(q) for q in ws.queues()]

    @app.get("/api/queues/{qid}")
    def queue(qid: int):
        try:
            return _batch(ws.queue(qid))
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
