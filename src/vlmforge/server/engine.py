"""One GPU worker for the UI: the base model loaded once, adapters swapped per request.

Requests queue FIFO on a single thread (the model is not thread-safe and the GB300 is shared,
GOALS §9). ``model="base"`` or an adapter ``.pt`` path; adapters are cached after first load.
Swapping = setting ``lora`` (the decoder is rebuilt by ``Moondream`` when it changes) and
loading / restoring the region head.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from pathlib import Path

from .. import paths

MAX_OBJECTS = 50


class Engine:
    def __init__(self, weights=None, tokenizer=None, batch_size: int = 16, lazy: bool = True):
        self.weights, self.tokenizer = weights or paths.WEIGHTS, tokenizer or paths.TOKENIZER
        self.batch_size = batch_size
        self.model = None
        self.base_head = None
        self.current = "base"
        self.adapters: dict[str, dict] = {}
        self.judge = None  # the Qwen3-VL judge, loaded on first use (GOALS §6)
        self.q: queue.Queue = queue.Queue()
        self.loaded_at: float | None = None
        self.busy = False
        self.error: str | None = None
        self.thread = threading.Thread(target=self._loop, daemon=True, name="forge-gpu")
        self.thread.start()
        if not lazy:
            self.submit(lambda m: None).result()

    # ------------------------------------------------------------------ worker
    def _load(self):
        import torch

        from ..model import Moondream

        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"model weights not found: {self.weights} (copy them from the PRO 6000, GOALS §3)"
            )
        t0 = time.time()
        self.model = Moondream(self.weights, self.tokenizer, batch_size=self.batch_size)
        self.base_head = {k: v.clone() for k, v in self.model.region.head_state_dict().items()}
        torch.cuda.empty_cache()
        self.loaded_at = time.time()
        print(f"engine: model loaded in {self.loaded_at - t0:.0f}s", flush=True)

    def _load_judge(self):
        from ..judge import Judge

        if not Path(paths.JUDGE_MODEL).exists():
            raise FileNotFoundError(f"judge model not found: {paths.JUDGE_MODEL}")
        t0 = time.time()
        self.judge = Judge(paths.JUDGE_MODEL)
        print(f"engine: judge loaded in {time.time() - t0:.0f}s", flush=True)

    def teacher_generate(self, image_path: str, focus: str = "", topics=None) -> dict:
        """Questions, answers and boxes for one image from the judge model (autogen)."""
        return self.submit(lambda j: j.generate(image_path, focus, topics), "judge").result()

    def judge_items(self, items: list[dict]) -> list[dict]:
        """Verdicts for ``items`` (``path``, ``skill``, ``prompt``, ``pred``), on the worker."""

        def run(j):
            return [j.judge(it["path"], it["skill"], it["prompt"], it["pred"]) for it in items]

        return self.submit(run, "judge").result()

    def _use(self, name: str):
        if name == self.current:
            return
        m = self.model
        if name == "base":
            m.lora = None
            m.region.load_head(self.base_head)
        else:
            if name not in self.adapters:
                from ..model import load_adapter

                self.adapters[name] = load_adapter(Path(name), m.device)
            a = self.adapters[name]
            m.lora = a["lora"]
            m.region.load_head(a["region_head"] or self.base_head)
        self.current = name

    def _loop(self):
        while True:
            fn, model_name, fut = self.q.get()
            if fut.set_running_or_notify_cancel() is False:
                continue
            self.busy = True
            try:
                if model_name == "judge":
                    if self.judge is None:
                        self._load_judge()
                    fut.set_result(fn(self.judge))
                    continue
                if self.model is None:
                    self._load()
                self._use(model_name)
                fut.set_result(fn(self.model))
            except Exception as e:  # noqa: BLE001 - reported to the caller, worker stays up
                self.error = f"{type(e).__name__}: {e}"
                fut.set_exception(e)
            finally:
                self.busy = False

    # ------------------------------------------------------------------ api
    def submit(self, fn, model: str = "base") -> Future:
        fut: Future = Future()
        self.q.put((fn, model, fut))
        return fut

    def pending(self) -> int:
        return self.q.qsize()

    def status(self) -> dict:
        mem = None
        try:
            import torch

            if torch.cuda.is_available():
                mem = round(torch.cuda.memory_allocated() / 2**30, 1)
        except Exception:  # noqa: BLE001
            pass
        return {
            "loaded": self.model is not None,
            "weights": str(self.weights),
            "weights_ok": Path(self.weights).exists(),
            "judge": str(paths.JUDGE_MODEL),
            "judge_ok": Path(paths.JUDGE_MODEL).exists(),
            "judge_loaded": self.judge is not None,
            "current": self.current,
            "adapters_loaded": sorted(self.adapters),
            "pending": self.pending(),
            "busy": self.busy,
            "gpu_mem_gb": mem,
            "last_error": self.error,
        }

    def infer(self, images: list, skill: str, prompt: str, model: str = "base") -> list[dict]:
        """Predictions for ``images`` (paths / arrays): detect -> {"boxes"}, point -> {"points"},
        query / caption -> {"text"}; coordinates normalised 0-1 as ``[x0, y0, x1, y1]``."""

        def run(m):
            if skill == "detect":
                out = m.detect(images, prompt, max_objects=MAX_OBJECTS)
                return [
                    {"boxes": [[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in bs]}
                    for bs in out
                ]
            if skill == "point":
                return [
                    {"points": [[p["x"], p["y"]] for p in ps]} for ps in m.point(images, prompt)
                ]
            if skill == "query":
                return [{"text": t} for t in m.query(images, prompt)]
            if skill == "caption":
                return [{"text": t} for t in m.caption(images, prompt or "normal")]
            raise ValueError(f"unknown skill {skill!r}")

        return self.submit(run, model).result()
