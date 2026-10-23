"""Step ④ "匯出到 Orin": a zip with the adapter, a README and a latency script (GOALS §7).

The adapter is the engine's own ``lora_state_dict`` layout (text MLPs only unless the run
trained the region head, which Photon cannot load), so on the Orin it is::

    pip install moondream
    import moondream as md
    m = md.vl(local=True, model="moondream3.1-9B-A2B", adapter="adapter.pt")

``bench_orin.py`` times base vs adapter on a folder of images, one line per skill with the
median milliseconds, and writes ``bench.json``. Scores are not recomputed on the Orin.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

README = """# {name} — Moondream 3.1 adapter

Photon 相容的 LoRA adapter（只掛文字 MLP，rank {rank}）。在 Orin AGX 上：

```bash
pip install moondream
python bench_orin.py --images ./some_frames --adapter adapter.pt
```

```python
import moondream as md
from PIL import Image

base = md.vl(local=True, model="moondream3.1-9B-A2B")
tuned = md.vl(local=True, model="moondream3.1-9B-A2B", adapter="adapter.pt")
img = Image.open("frame.jpg")
print(tuned.detect(img, "person")["objects"])
print(tuned.query(img, "Is anyone without a helmet?")["answer"])
```

訓練：{rows} 筆、{steps} 步、lr {lr}；held-out 分數見 `scores.json`。
若 `adapter.pt` 內含 `region_head`（進階選項），Photon 載不了，只能在 VLM-Forge 跑。
"""

BENCH = r'''"""Latency on the Orin: base vs adapter, per skill, median ms (no scores here).

    python bench_orin.py --images ./frames --adapter adapter.pt [--n 20]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--adapter", default="adapter.pt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--object", default="person")
    ap.add_argument("--question", default="Is anyone without a helmet?")
    a = ap.parse_args()
    import moondream as md

    exts = (".jpg", ".jpeg", ".png")
    files = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in exts)
    files = files[: a.n]
    out = {}
    for label, kw in (("base", {}), ("adapter", {"adapter": a.adapter})):
        m = md.vl(local=True, model="moondream3.1-9B-A2B", **kw)
        t = {"detect": [], "point": [], "query": [], "caption": []}
        for p in files:
            img = Image.open(p)
            for skill, fn in (
                ("detect", lambda: m.detect(img, a.object)),
                ("point", lambda: m.point(img, a.object)),
                ("query", lambda: m.query(img, a.question)),
                ("caption", lambda: m.caption(img, length="normal")),
            ):
                t0 = time.perf_counter()
                fn()
                t[skill].append((time.perf_counter() - t0) * 1000)
        out[label] = {k: round(statistics.median(v), 1) for k, v in t.items() if v}
        print(label, out[label], flush=True)
    Path("bench.json").write_text(json.dumps({"images": len(files), **out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def make_bundle(run_dir: Path, name: str, meta: dict) -> Path:
    """``runs/<name>/<name>-orin.zip`` with adapter.pt, README.md, bench_orin.py, scores.json."""
    adapter = run_dir / f"{name}.pt"
    if not adapter.exists():
        raise FileNotFoundError(f"no adapter at {adapter}")
    zpath = run_dir / f"{name}-orin.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(adapter, "adapter.pt")
        z.writestr(
            "README.md",
            README.format(
                name=name,
                rank=meta.get("rank", "?"),
                rows=meta.get("rows", "?"),
                steps=meta.get("steps", "?"),
                lr=meta.get("lr", "?"),
            ),
        )
        z.writestr("bench_orin.py", BENCH)
        z.writestr(
            "scores.json",
            json.dumps({"evals": meta.get("evals", {}), "gate": meta.get("gate")}, indent=1),
        )
    return zpath
