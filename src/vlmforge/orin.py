"""Step ③ "匯出到 Orin": a zip with the adapter, a loader, a README and a latency script
(GOALS §3, §7).

The adapter is the engine's own ``lora_state_dict`` layout (text MLPs only unless the run
trained the region head, which Photon cannot load). Checked against kestrel 0.9.2 /
moondream 2.6.1 (2026-10-23):

* kestrel builds ``LoRA(text=TextLoRA(...))`` and registers the text part as ``text_lora``,
  so the state-dict keys are ``text_lora.dense.{i}.{up_a,up_b,down_a,down_b}`` and
  ``text_lora.moe.{j}.*`` — exactly what :meth:`vlmforge.model.text.TextLoRA.engine_state_dict`
  writes. Rank is read from ``dense.0.up_a.shape[0]`` and must divide by 8; there is no alpha
  (ours is folded into B).
* The ``moondream`` client only understands adapters as cloud ids (``finetune_id@step``)
  fetched from the Moondream API, and kestrel ships only that cloud ``AdapterProvider``.
  There is no ``adapter="file.pt"`` argument. Local adapters therefore go through
  ``local_adapter.py`` in the bundle: a small provider implementing the same protocol
  (``config()`` + ``get(adapter_id)``) that reads a ``.pt`` from disk, plus a sync wrapper so
  the Orin script reads like the client.

``bench_orin.py`` times base vs adapter on a folder of images, one line per skill with the
median milliseconds, and writes ``bench.json``. Scores are not recomputed on the Orin.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

MODEL_ID = "moondream3.1-9B-A2B"  # kestrel's registered name for the 3.1 weights

README = """# {name} — Moondream 3.1 adapter

Photon（kestrel）相容的 LoRA adapter：只掛文字 MLP，rank {rank}，key 版面
`text_lora.dense.*` / `text_lora.moe.*`，alpha 已折進 B 矩陣。

## 在 Orin AGX 上

```bash
pip install moondream            # 會帶 kestrel（有 Jetson aarch64 kernel）
python bench_orin.py --images ./some_frames --adapter adapter.pt
```

```python
from PIL import Image
from local_adapter import Tuned

m = Tuned("adapter.pt")                  # 第一次載 base 約需幾十秒
img = Image.open("frame.jpg")
print(m.detect(img, "person")["objects"])
print(m.query(img, "Is anyone without a helmet?")["answer"])
print(m.base.detect(img, "person")["objects"])   # 同一個 engine 跑 base 做對照
```

為什麼不用 `md.vl(local=True, …)`：官方 client 的 adapter 只認雲端 id
（`finetune_id@step`），不吃本機檔案；`local_adapter.py` 用 kestrel 的 `AdapterProvider`
介面直接從磁碟載，模型本體與推論路徑都還是 Photon 的。

## 注意

- kestrel 啟動時若印出 `MoE LoRA adapters are unavailable in this kernel runtime`，表示這個
  平台的 kernel 不支援 MoE 層的 LoRA，adapter 會被拒絕、只剩 base——把那段 log 貼回 VLM-Forge。
- 若 `adapter.pt` 內含 `region_head`（進階選項），Photon 載不了，只能在 VLM-Forge 跑。

訓練：{rows} 筆、{steps} 步、lr {lr}；held-out 分數見 `scores.json`。
"""

LOCAL_ADAPTER = r'''"""Load a VLM-Forge adapter (.pt) into Photon / kestrel from a local file.

kestrel's engine takes any object with ``config() -> {"max_lora_rank"}`` and
``get(adapter_id) -> kestrel LoRA``; the id we use is the file path. ``Tuned`` wraps the
async engine so a script can call ``detect / point / query / caption`` synchronously.
Written against kestrel 0.9.2.
"""

from __future__ import annotations

import asyncio
import io
import threading
from collections import OrderedDict
from pathlib import Path

import torch

MODEL_ID = "%(model_id)s"


class LocalAdapterProvider:
    """``AdapterProvider`` that reads ``{"lora_state_dict", "rank"}`` files from disk."""

    def __init__(self, text_config, device, dtype=torch.bfloat16, max_lora_rank: int = 32):
        self._text_config, self._device, self._dtype = text_config, device, dtype
        self._max_lora_rank = max_lora_rank
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    def config(self) -> dict:
        return {"max_lora_rank": self._max_lora_rank}

    def get(self, adapter: str):
        from kestrel.models.moondream.lora import LoRA, TextLoRA, TextLoRAConfig

        key = str(Path(adapter).resolve())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ckpt = torch.load(key, map_location="cpu", weights_only=True)
        sd = ckpt["lora_state_dict"]
        rank = int(ckpt.get("rank") or sd["text_lora.dense.0.up_a"].shape[0])
        if rank > self._max_lora_rank:
            raise ValueError(f"adapter rank {rank} > max_lora_rank {self._max_lora_rank}")
        if ckpt.get("region_head"):
            raise ValueError("this adapter trained the region head; Photon cannot load it")
        text = TextLoRA(self._text_config, TextLoRAConfig(rank=rank), dtype=self._dtype)
        lora = LoRA(text=text)
        missing, unexpected = lora.load_state_dict(sd, strict=False)
        bad = [k for k in missing if k.startswith("text_lora.")]
        if unexpected or bad:
            raise ValueError(
                f"adapter keys do not match kestrel: missing={bad[:3]} unexpected={unexpected[:3]}"
            )
        lora.to(self._device)
        self._cache[key] = lora
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return lora


class _Loop:
    """One background event loop shared by every ``Tuned`` instance."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


_ENGINE = None


def engine(device: str = "cuda"):
    """The kestrel engine for MODEL_ID with the local adapter provider (created once)."""
    global _ENGINE
    if _ENGINE is None:
        from kestrel import InferenceEngine
        from kestrel.config import RuntimeConfig
        from kestrel.models.moondream.config import load_config

        cfg = RuntimeConfig(model=MODEL_ID, device=device)
        provider = LocalAdapterProvider(
            load_config().text, torch.device(device), cfg.resolved_dtype()
        )
        lp = _Loop()
        eng = lp.run(InferenceEngine.create(cfg, adapter_provider=provider))
        _ENGINE = (eng, lp)
    return _ENGINE


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


class Tuned:
    """Sync ``detect / point / query / caption`` on base (``adapter=None``) or an adapter file.

    Returns the same dict shapes as the ``moondream`` client.
    """

    def __init__(self, adapter: str | None = None, device: str = "cuda"):
        self.adapter = str(Path(adapter).resolve()) if adapter else None
        eng, self._lp = engine(device)
        self._m = eng.model(MODEL_ID)

    @property
    def base(self) -> "Tuned":
        return Tuned(None) if self.adapter else self

    def _settings(self):
        return {"adapter": self.adapter} if self.adapter else None

    def detect(self, image, obj: str) -> dict:
        r = self._lp.run(self._m.detect(image=_png(image), object=obj, settings=self._settings()))
        return {"objects": r.output["objects"]}

    def point(self, image, obj: str) -> dict:
        r = self._lp.run(self._m.point(image=_png(image), object=obj, settings=self._settings()))
        return {"points": r.output["points"]}

    def query(self, image, question: str) -> dict:
        r = self._lp.run(
            self._m.query(
                image=_png(image), question=question, stream=False, settings=self._settings()
            )
        )
        return {"answer": r.output["answer"]}

    def caption(self, image, length: str = "normal") -> dict:
        r = self._lp.run(
            self._m.caption(
                image=_png(image), length=length, stream=False, settings=self._settings()
            )
        )
        return {"caption": r.output["caption"]}
'''

BENCH = r'''"""Latency on the Orin: base vs adapter, per skill, median ms (no scores here).

    python bench_orin.py --images ./frames --adapter adapter.pt [--n 20]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from local_adapter import Tuned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--adapter", default="adapter.pt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--object", default="person")
    ap.add_argument("--question", default="Is anyone without a helmet?")
    a = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png")
    files = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in exts)
    files = files[: a.n]
    out = {}
    for label, m in (("base", Tuned(None)), ("adapter", Tuned(a.adapter))):
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

BUNDLE_FILES = ("adapter.pt", "local_adapter.py", "README.md", "bench_orin.py", "scores.json")


def local_adapter_source() -> str:
    """``local_adapter.py`` as shipped in the bundle (unit-tested for syntax)."""
    return LOCAL_ADAPTER % {"model_id": MODEL_ID}


def make_bundle(run_dir: Path, name: str, meta: dict) -> Path:
    """``runs/<name>/<name>-orin.zip`` with the files in :data:`BUNDLE_FILES`."""
    adapter = run_dir / f"{name}.pt"
    if not adapter.exists():
        raise FileNotFoundError(f"no adapter at {adapter}")
    zpath = run_dir / f"{name}-orin.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(adapter, "adapter.pt")
        z.writestr("local_adapter.py", local_adapter_source())
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
