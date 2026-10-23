"""Training / eval rows and the manifests built from Data Core releases (GOALS §4, §7).

Row format (one JSON object per line, coordinates normalised to 0-1; shared with
SyncVision so scores are comparable):

    {"image": "...", "skill": "query",   "question": "...", "answer": "..."}
    {"image": "...", "skill": "caption", "length": "short|normal|long", "answer": "..."}
    {"image": "...", "skill": "detect",  "object": "person", "boxes": [[x0, y0, x1, y1]]}
    {"image": "...", "skill": "point",   "object": "person", "points": [[x, y]]}

Optional provenance fields: ``source`` (the macro group with ``object``), ``id``,
``label_origin``, ``reviewer``, ``date``.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

GOLD_QUESTIONS = [("people", "How many people are there?"), ("squat", "Is anyone squatting?")]


def read_rows(paths: list[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def write_rows(path: str | Path, rows: list[dict], ensure_ascii: bool = True) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=ensure_ascii) + "\n")


def split_rows(rows: list[dict], val: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Hold out ``val`` images (one photograph never spans both halves)."""
    images = sorted({r["image"] for r in rows})
    random.Random(seed).shuffle(images)
    held = set(images[:val])
    return [r for r in rows if r["image"] not in held], [r for r in rows if r["image"] in held]


def mixture(entries: list, seed: int) -> list[dict]:
    """Manifest paths, or ``{path, weight, optional}``: weight 2.5 = every row twice plus a
    random half; optional entries whose file does not exist yet are skipped."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for e in entries:
        path, w = (e, 1.0) if isinstance(e, str) else (e["path"], float(e.get("weight", 1.0)))
        if not isinstance(e, str) and e.get("optional") and not Path(path).exists():
            print(f"mixture: optional {path} not built yet, skipped", flush=True)
            continue
        part = read_rows([path])
        rows += part * int(w)
        frac = w - int(w)
        if frac:
            rows += rng.sample(part, round(frac * len(part)))
    return rows


def image_key(path: str) -> str:
    """Release-independent image identity: the path below a release's ``data/files/``."""
    return path.split("/data/files/", 1)[-1]


# ------------------------------------------------------------------ held-out release
def crowdhuman_rows(odgt: Path, images: dict[str, str], n: int = 500, max_count: int = 20):
    """CrowdHuman val rows: ``images`` maps ID -> path for (at least) the sampled IDs."""
    recs = [json.loads(line) for line in open(odgt)]
    random.Random(0).shuffle(recs)
    rows = []
    for r in recs[:n]:
        img = images[r["ID"]]
        w, h = Image.open(img).size
        boxes = []
        for g in r["gtboxes"]:
            if g.get("tag") != "person" or g.get("extra", {}).get("ignore", 0):
                continue
            x, y, bw, bh = g["vbox"]
            x0, y0 = max(0.0, x / w), max(0.0, y / h)
            x1, y1 = min(1.0, (x + bw) / w), min(1.0, (y + bh) / h)
            if x1 > x0 and y1 > y0:
                boxes.append([x0, y0, x1, y1])
        base = {"image": img, "source": "crowdhuman", "id": r["ID"]}
        rows.append({**base, "skill": "detect", "object": "person", "boxes": boxes})
        if boxes:
            rows.append(
                {
                    **base,
                    "skill": "point",
                    "object": "person",
                    "boxes": boxes,
                    "points": [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes],
                }
            )
        if len(boxes) <= max_count:
            rows.append(
                {
                    **base,
                    "skill": "query",
                    "question": "How many people are there?",
                    "answer": str(len(boxes)),
                }
            )
    return rows


def heldout_manifests(data: Path) -> dict[str, list[dict]]:
    """The three held-out suites from a ``syncvision-heldout-eval`` release's ``data/``,
    row for row what SyncVision's ``make_heldout_eval.py`` writes (GOALS §7: shared KPI).
    Paths are kept as given, never resolved (release files are DVC cache symlinks)."""

    def path(f: str) -> str:
        return os.path.abspath(data / f)

    by = defaultdict(list)
    for line in open(data / "manifest.jsonl"):
        r = json.loads(line)
        by[r["input"].split("/")[-1].split("@")[0]].append(r)
    outs: dict[str, list[dict]] = {}

    ch = by["crowdhuman-20261022"]
    odgt = next((data / "inputs").glob("*crowdhuman*/raw/val/annotation_val.odgt"))
    outs["heldout-crowdhuman-val"] = crowdhuman_rows(
        odgt, {r["ID"]: path(r["image"]) for r in ch}, n=len(ch)
    )

    gold = []
    for r in sorted(by["people-posture-gold-20261022"], key=lambda r: r["stamp_ns"]):
        answers = {"people": str(r["people"]), "squat": "yes" if r["squat"] else "no"}
        for key, q in GOLD_QUESTIONS:
            gold.append(
                {
                    "image": path(r["image"]),
                    "skill": "query",
                    "question": q,
                    "answer": answers[key],
                    "source": "2026-10-07-people-posture",
                    "id": r["id"],
                    "label_origin": "human",
                    "reviewer": r["reviewer"],
                    "date": r["date"],
                }
            )
    outs["heldout-people-posture-gold"] = gold

    cctv = []
    for r in sorted(by["factory-cctv-labels-20261022"], key=lambda r: r["id"]):
        base = {
            "image": path(r["image"]),
            "source": "cctv",
            "id": r["id"].split("/", 1)[1],
            "label_origin": "human",
            "prefill": "syncvlm",
            "reviewer": r["reviewer"],
            "date": r["date"],
        }
        bs, obj = r["boxes"], r["object"]
        cctv.append({**base, "skill": "detect", "object": obj, "boxes": bs})
        if bs:
            cctv.append(
                {
                    **base,
                    "skill": "point",
                    "object": obj,
                    "boxes": bs,
                    "points": [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in bs],
                }
            )
        if obj == "person" and len(bs) <= 30:
            cctv.append(
                {
                    **base,
                    "skill": "query",
                    "question": "How many people are there?",
                    "answer": str(len(bs)),
                }
            )
    outs["heldout-cctv"] = cctv
    return outs


def build_heldout(release: Path, out: Path) -> dict[str, int]:
    """Write the held-out manifests to ``out`` and return row counts per suite."""
    outs = heldout_manifests(release / "data" if (release / "data").is_dir() else release)
    counts = {}
    for name, rows in outs.items():
        write_rows(out / f"{name}.jsonl", rows, ensure_ascii=name != "heldout-people-posture-gold")
        counts[name] = len(rows)
    return counts
