"""Scoring (SyncVision's metrics, GOALS §7) and the ``forge eval`` runner.

detect: F1 at IoU 0.5 (greedy matching); point: a point hits the box it lies in (rows with
``boxes``) or lands within radius 0.05 of a GT point; query: exact match (numbers by value,
yes/no by the first word). ``<skill>/macro`` = mean over (source, object) groups, the KPI
aggregate. Groups with only negatives score the share answered with no false positive.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred: list, gt: list, thr: float = 0.5) -> tuple[int, int, int]:
    used, tp = set(), 0
    for p in pred:
        best, bj = 0.0, -1
        for j, t in enumerate(gt):
            if j not in used and (v := iou(p, t)) > best:
                best, bj = v, j
        if best >= thr:
            used.add(bj)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def match_points(pred: list, gt: list, radius: float = 0.05) -> tuple[int, int, int]:
    used, tp = set(), 0
    for p in pred:
        best, bj = radius, -1
        for j, t in enumerate(gt):
            d = ((p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
            if j not in used and d <= best:
                best, bj = d, j
        if bj >= 0:
            used.add(bj)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def match_points_in_boxes(pred: list, boxes: list) -> tuple[int, int, int]:
    """A point hits a box it lies in; each box takes at most one point (smallest box first)."""
    order = sorted(
        range(len(boxes)), key=lambda j: (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
    )
    used, tp = set(), 0
    for p in pred:
        for j in order:
            b = boxes[j]
            if j not in used and b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]:
                used.add(j)
                tp += 1
                break
    return tp, len(pred) - tp, len(boxes) - tp


def f1(tp: int, fp: int, fn: int) -> float:
    """F1; a group with no objects and no predictions (all negatives answered right) is 1."""
    if tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn)


def number(text: str) -> int | None:
    m = re.search(r"\d+", text)
    if m:
        return int(m.group())
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    for i, w in enumerate(words):
        if re.search(rf"\b{w}\b", text.lower()):
            return i
    return None


def score(rows: list[dict], preds: list) -> dict:
    """Metrics per skill and per skill:object. preds[i] is boxes, points or text for rows[i]."""
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    exact: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    neg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    items = []
    for r, pred in zip(rows, preds, strict=True):
        what = r.get("object") or r.get("question", "")[:40]
        keys = [
            r["skill"],
            f"{r['skill']}:{what}",
            f"{r['skill']}@{r.get('source', '-')}:{what}",  # macro group (GOALS §7)
        ]
        if r["skill"] in ("detect", "point"):
            if r["skill"] == "detect":
                m = match_boxes(pred, r["boxes"])
            elif "boxes" in r:  # points derived from boxes: hit = inside the box
                m = match_points_in_boxes(pred, r["boxes"])
            else:
                m = match_points(pred, r["points"])
            for k in keys:
                agg[k] = [a + b for a, b in zip(agg[k], m, strict=True)]
                if m[0] + m[2] == 0:  # a negative item: right when nothing is predicted
                    neg[k][0] += m[1] == 0
                    neg[k][1] += 1
            items.append({"image": r["image"], "object": r["object"], "pred": pred, "tp_fp_fn": m})
        else:
            gt = str(r["answer"]).strip().lower()
            n = number(pred) if gt.isdigit() else None
            if gt.isdigit():
                ok = n == int(gt)
            elif gt in ("yes", "no"):  # yes/no: the answer's first word decides
                first = re.findall(r"[a-z]+", pred.lower())[:1]
                ok = first == [gt]
            else:
                ok = pred.strip().lower().rstrip(".") == gt
            for k in keys:
                exact[k][0] += ok
                exact[k][1] += 1
            items.append(
                {
                    "image": r["image"],
                    "question": r.get("question"),
                    "pred": pred,
                    "answer": r["answer"],
                    "ok": ok,
                }
            )

    def group(k: str, v: list[int]) -> float:
        # a group without any object is scored by its negatives: share with no false positive
        return neg[k][0] / neg[k][1] if v[0] + v[2] == 0 and neg[k][1] else f1(*v)

    out = {f"{k}/f1": group(k, v) for k, v in sorted(agg.items())}
    out.update({f"{k}/exact": c / max(1, n) for k, (c, n) in sorted(exact.items())})
    out.update({f"{k}/n": sum(v[:1]) + v[2] for k, v in agg.items() if ":" not in k})
    # macro: mean over (source, object) groups, the KPI aggregate (GOALS §7)
    for skill in {k.split("@")[0] for k in list(agg) + list(exact) if "@" in k}:
        f = [group(k, v) for k, v in agg.items() if k.startswith(f"{skill}@")]
        e = [c / max(1, n) for k, (c, n) in exact.items() if k.startswith(f"{skill}@")]
        out[f"{skill}/macro"] = sum(f + e) / max(1, len(f + e))
    return {"metrics": out, "items": items}


def macro(metrics: dict) -> dict[str, float]:
    return {k.split("/")[0]: v for k, v in metrics.items() if k.endswith("/macro")}


# ------------------------------------------------------------------ running a model
def predict_rows(model, rows: list[dict]) -> list:
    """Greedy predictions for ``rows`` (any mix of skills), batched per skill."""
    preds: list = [None] * len(rows)
    by_skill = defaultdict(list)
    for i, r in enumerate(rows):
        by_skill[r["skill"]].append(i)
    for skill, idx in by_skill.items():
        imgs = [rows[i]["image"] for i in idx]
        if skill == "detect":
            out = model.detect(imgs, [rows[i]["object"] for i in idx])
            out = [[[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in bs] for bs in out]
        elif skill == "point":
            out = model.point(imgs, [rows[i]["object"] for i in idx])
            out = [[[p["x"], p["y"]] for p in ps] for ps in out]
        elif skill == "caption":
            lengths = {rows[i].get("length", "normal") for i in idx}
            out = [None] * len(idx)
            for ln in lengths:
                sub = [j for j, i in enumerate(idx) if rows[i].get("length", "normal") == ln]
                for j, text in zip(sub, model.caption([imgs[j] for j in sub], ln), strict=True):
                    out[j] = text
        else:
            out = model.query(imgs, [rows[i]["question"] for i in idx])
        for i, p in zip(idx, out, strict=True):
            preds[i] = p
    return preds


def evaluate(manifest: Path, out: Path, adapter: Path | None = None, batch: int = 64) -> dict:
    """Score the base model (or base + adapter) on a manifest and write the report."""
    import torch

    from . import paths
    from .model import Moondream, load_adapter

    rows = [json.loads(line) for line in open(manifest)]
    extra = load_adapter(adapter, "cuda") if adapter else {}
    model = Moondream(paths.WEIGHTS, paths.TOKENIZER, batch_size=batch, **extra)
    with torch.no_grad():
        preds = predict_rows(model, rows)
    res = score(rows, preds)
    res["model"] = "syncvlm" if adapter is None else f"syncvlm+{Path(adapter).stem}"
    res["manifest"] = str(manifest)
    if adapter:
        res["adapter"] = str(adapter)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ picking a checkpoint
def pick_checkpoint(run_dir: Path, name: str, manifests: list[Path], batch: int = 32) -> dict:
    """Small datasets overfit, so the last step is not always the best one (GOALS §6). Score
    every ``ckpt-<step>.pt`` of a run on the datasets' validation splits with one model load
    (mean of the skills' macro scores), keep the best as ``<name>.pt`` and delete the rest.
    Without validation rows the lowest ``val/loss`` in ``log.jsonl`` decides; without that,
    the last step. Writes and returns ``checkpoints.json``."""
    import shutil

    ckpts = sorted(run_dir.glob("ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    steps = [int(p.stem.split("-")[1]) for p in ckpts]
    out: dict = {"steps": steps, "scores": {}, "by": "last", "best": steps[-1] if steps else None}
    manifests = [m for m in manifests if m.exists()]
    if len(ckpts) > 1 and manifests:
        import torch

        from . import paths
        from .model import Moondream, load_adapter

        rows = [json.loads(line) for m in manifests for line in open(m)]
        model = Moondream(paths.WEIGHTS, paths.TOKENIZER, batch_size=batch)
        base_head = {k: v.clone() for k, v in model.region.head_state_dict().items()}
        with torch.no_grad():
            for step, ck in zip(steps, ckpts, strict=True):
                a = load_adapter(ck, model.device)
                model.lora = a["lora"]
                model.region.load_head(a["region_head"] or base_head)
                m = macro(score(rows, predict_rows(model, rows))["metrics"])
                out["scores"][str(step)] = round(sum(m.values()) / len(m), 4) if m else 0.0
                print(f"checkpoint {step}: {out['scores'][str(step)]}", flush=True)
        out["by"] = "val"
        out["best"] = max(steps, key=lambda s: (out["scores"][str(s)], s))  # tie → later
    elif len(ckpts) > 1 and (run_dir / "log.jsonl").exists():
        val = {}
        for line in open(run_dir / "log.jsonl"):
            rec = json.loads(line)
            if "val/loss" in rec and rec["step"] in steps:
                val[rec["step"]] = rec["val/loss"]
        if val:
            out["scores"] = {str(k): round(v, 4) for k, v in val.items()}
            out["by"] = "val_loss"
            out["best"] = min(val, key=lambda s: (val[s], -s))
    if out["best"] is not None:
        shutil.copyfile(run_dir / f"ckpt-{out['best']}.pt", run_dir / f"{name}.pt")
        for ck in ckpts:
            ck.unlink()
    (run_dir / "checkpoints.json").write_text(json.dumps(out, indent=1))
    return out
