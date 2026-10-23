"""Exports from the workspace (GOALS §7).

``export_train`` writes reviewed items as training / eval rows (``vlmforge.data`` format) that
experimental runs read directly. ``export_incoming`` writes a Data Core intake bundle under
``datasets/_incoming/<name>/`` for the Data Core owner to register as a source (the new
frames) plus a labels layer (``items.jsonl``, ``label_origin: human``); nothing here touches a
Data Core clone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml

from .. import paths
from ..data import write_rows
from .workspace import Workspace


def export_train(ws: Workspace, queue_ids: list[int] | None, name: str) -> dict:
    rows = ws.reviewed_rows(queue_ids)
    out = paths.WORKSPACE / "export" / f"{name}.jsonl"
    write_rows(out, rows)
    skills = {}
    for r in rows:
        skills[r["skill"]] = skills.get(r["skill"], 0) + 1
    return {"path": str(out), "rows": len(rows), "skills": skills}


def export_incoming(ws: Workspace, queue_ids: list[int] | None, name: str, out_dir: Path) -> dict:
    """Bundle: ``raw/<source>/<sha>.<ext>`` (workspace images only; release images are
    referenced by their release path), ``items.jsonl``, ``files.sha256`` and draft
    ``card.yaml`` / ``README.md`` for the labels layer."""
    rows = ws.reviewed_rows(queue_ids)
    root = out_dir / name
    (root / "raw").mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    shas: list[str] = []
    items = []
    for r in rows:
        src = Path(r["image"])
        if src.is_relative_to(ws.images_dir):
            rel = copied.get(str(src))
            if rel is None:
                sub = r["source"].replace(":", "_")
                dst = root / "raw" / sub / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                rel = f"raw/{sub}/{src.name}"
                copied[str(src)] = rel
                shas.append(f"{hashlib.sha256(src.read_bytes()).hexdigest()}  {rel}")
            image = rel
        else:
            image = str(src)
        items.append({**r, "image": image})
    write_rows(root / "items.jsonl", items)
    (root / "files.sha256").write_text("\n".join(shas) + ("\n" if shas else ""))
    heads = sorted({"object" if r["skill"] in ("detect", "point") else r["skill"] for r in rows})
    card = {
        "schema_version": 1,
        "id": f"vlm-corpus/labels/{name}",
        "kind": "labels",
        "version": "v1",
        "owner": "PSheon",
        "status": "draft",
        "tier": "T2",
        "storage": "restricted",
        "first_party": True,
        "pii": ["faces"],
        "label_origin": {h: "human" for h in heads},
        "provenance": {"tool": "VLM-Forge forge export", "items": len(items)},
    }
    (root / "card.draft.yaml").write_text(yaml.safe_dump(card, sort_keys=False))
    (root / "README.draft.md").write_text(
        f"# {name}\n\n{len(items)} items reviewed in VLM-Forge (human, prefilled by the model;"
        f" `prefill` names the model, `changed` whether the reviewer edited it).\n"
        f"Images under `raw/` are new captures to register as a source; other `image` values"
        f" point at an existing release.\n"
    )
    return {"path": str(root), "items": len(items), "images": len(copied)}


def manifest_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            s = json.loads(line)["skill"]
            counts[s] = counts.get(s, 0) + 1
    return counts
