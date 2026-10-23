"""Step ③/④ of the UI: train an adapter on everything reviewed so far, then score it on
the held-out suites and compare with base (GOALS §6).

``Pipeline.start(name, steps, lr)`` writes ``runs/<name>/config.yaml`` (the reviewed rows
exported to ``workspace/export/<name>.jsonl``, held-out images excluded), runs
``python -m vlmforge.train`` as a subprocess, then ``forge eval`` per suite, then the gate.
Progress lives in ``runs/<name>/pipeline.json`` so the page can poll it and a restart of the
server does not lose it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from .. import paths
from ..gate import combined_macro, gate
from .export import export_train
from .workspace import Workspace

DEFAULT_STEPS, DEFAULT_LR, DEFAULT_RANK = 500, 2.0e-4, 16


def default_name() -> str:
    return datetime.now().strftime("ft-%Y%m%d-%H%M")


def write_config(name: str, train_jsonl: Path, n_images: int, steps: int, lr: float, rank: int):
    cfg = {
        "name": name,
        "out": str(paths.RUNS / name),
        "seed": 0,
        "rank": rank,
        "alpha": 2 * rank,
        "region_head": True,
        "gpu_mem_gb": 60,
        "data": {
            "train": [{"path": str(train_jsonl), "weight": 1.0}],
            "exclude": [
                str(paths.MANIFESTS / f"{s}.jsonl")
                for s in paths.HELDOUT_SUITES
                if (paths.MANIFESTS / f"{s}.jsonl").exists()
            ],
            "val_images": max(4, min(50, n_images // 20)),
            "max_objects": 50,
            "max_text": 384,
            "workers": 16,
        },
        "replay": {
            "ratio": 0.5,
            "images": max(16, min(256, n_images)),
            "per_image_text": 2,
            "per_image_spatial": 1,
            "batch": 32,
        },
        "optim": {
            "lr": lr,
            "warmup": max(5, steps // 20),
            "steps": steps,
            "batch": 8,
            "min_lr_ratio": 0.1,
            "clip": 1.0,
        },
        "log_every": 10,
        "eval_every": max(50, steps // 4),
        "eval_batches": 4,
        "save_every": max(50, steps // 4),
    }
    out = paths.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out / "config.yaml"


class Pipeline:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------ status
    @staticmethod
    def _state_path(name: str) -> Path:
        return paths.RUNS / name / "pipeline.json"

    def status(self, name: str) -> dict | None:
        p = self._state_path(name)
        if not p.exists():
            return None
        st = json.loads(p.read_text())
        log = paths.RUNS / name / "log.jsonl"
        curve, last = [], None
        if log.exists():
            for line in open(log):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                curve.append(
                    {"step": rec["step"], "loss": rec.get("loss"), "val": rec.get("val/loss")}
                )
                last = rec
        st["curve"], st["last"] = curve, last
        t = self.threads.get(name)
        st["alive"] = bool(t and t.is_alive())
        return st

    def list(self) -> list[dict]:
        out = []
        for p in sorted(paths.RUNS.glob("*/pipeline.json")):
            st = self.status(p.parent.name)
            if st:
                out.append(st)
        return sorted(out, key=lambda s: s.get("started", ""), reverse=True)

    def _write(self, name: str, st: dict) -> None:
        self._state_path(name).write_text(json.dumps(st, indent=1, ensure_ascii=False))

    # ------------------------------------------------------------------ run
    def start(self, name: str, steps: int, lr: float, rank: int = DEFAULT_RANK) -> dict:
        if any(t.is_alive() for t in self.threads.values()):
            raise RuntimeError("a training pipeline is already running")
        if (paths.RUNS / name).exists():
            raise RuntimeError(f"run {name} exists")
        exp = export_train(self.ws, None, name)
        if exp["rows"] == 0:
            raise RuntimeError("nothing to train on: review some data first")
        n_images = len({json.loads(line)["image"] for line in open(exp["path"])})
        cfg = write_config(name, Path(exp["path"]), n_images, steps, lr, rank)
        st = {
            "name": name,
            "state": "running",
            "phase": "training",
            "started": datetime.now().isoformat(timespec="seconds"),
            "steps": steps,
            "lr": lr,
            "rank": rank,
            "rows": exp["rows"],
            "skills": exp["skills"],
            "images": n_images,
            "config": str(cfg),
            "evals": {},
            "gate": None,
            "error": None,
        }
        self._write(name, st)
        t = threading.Thread(target=self._run, args=(name, st, cfg), daemon=True)
        self.threads[name] = t
        t.start()
        return st

    def _cmd(self, *args: str) -> list[str]:
        base = [sys.executable, "-W", "ignore", "-m", "vlmforge.cli", *args]
        return ["numactl", "--membind=0", *base] if shutil.which("numactl") else base

    def _run(self, name: str, st: dict, cfg: Path) -> None:
        run_dir = paths.RUNS / name
        log = open(run_dir / "train.log", "a")
        try:
            rc = subprocess.call(
                self._cmd("train", str(cfg)), cwd=paths.ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            if rc != 0:
                raise RuntimeError(f"training exited with {rc} (see runs/{name}/train.log)")
            adapter = run_dir / f"{name}.pt"
            if not adapter.exists():
                raise RuntimeError("training produced no adapter")
            for suite in paths.HELDOUT_SUITES:
                manifest = paths.MANIFESTS / f"{suite}.jsonl"
                if not manifest.exists():
                    continue
                st["phase"] = f"eval:{suite}"
                self._write(name, st)
                rc = subprocess.call(
                    self._cmd("eval", suite, "--adapter", str(adapter), "--batch", "32"),
                    cwd=paths.ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                rep = run_dir / f"eval-{name}-{suite}.json"
                if rc == 0 and rep.exists():
                    m = json.loads(rep.read_text())["metrics"]
                    st["evals"][suite] = {
                        k.split("/")[0]: v for k, v in m.items() if k.endswith("/macro")
                    }
                    self._write(name, st)
            base_reports = [paths.BASELINES / f"base-{s}.json" for s in st["evals"]]
            base_reports = [p for p in base_reports if p.exists()]
            if base_reports:
                adapter_reports = [run_dir / f"eval-{name}-{s}.json" for s in st["evals"]]
                st["gate"] = gate(
                    combined_macro(base_reports), combined_macro(adapter_reports), "multi"
                )
            st["phase"], st["state"] = "done", "done"
        except Exception as e:  # noqa: BLE001 - shown on the page
            st["state"], st["error"] = "error", f"{type(e).__name__}: {e}"
        finally:
            st["finished"] = datetime.now().isoformat(timespec="seconds")
            self._write(name, st)
            log.close()
            time.sleep(0)
