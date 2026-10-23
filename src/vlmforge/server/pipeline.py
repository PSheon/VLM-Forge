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
from ..data import write_rows
from ..gate import combined_macro, gate
from .export import export_train
from .workspace import Workspace

DEFAULT_STEPS, DEFAULT_LR, DEFAULT_RANK = 500, 2.0e-4, 16


def default_name() -> str:
    return datetime.now().strftime("ft-%Y%m%d-%H%M")


def write_config(
    name: str,
    train_jsonl: Path,
    n_images: int,
    steps: int,
    lr: float,
    rank: int,
    region_head: bool = False,
    val_jsonl: list[Path] | None = None,
):
    """The YAML ``forge train`` reads. ``val_jsonl``: the datasets' own validation rows; with
    them the trainer holds nothing else out (a 2-image run once lost both images to the
    inner split and crashed), without them it holds out n_images // 20 (0 for tiny sets)."""
    cfg = {
        "name": name,
        "out": str(paths.RUNS / name),
        "seed": 0,
        "rank": rank,
        "alpha": 2 * rank,
        "region_head": region_head,
        "gpu_mem_gb": 60,
        "data": {
            "train": [{"path": str(train_jsonl), "weight": 1.0}],
            "exclude": [
                str(paths.MANIFESTS / f"{s}.jsonl")
                for s in paths.HELDOUT_SUITES
                if (paths.MANIFESTS / f"{s}.jsonl").exists()
            ],
            # without the datasets' own val rows, hold out ~10 % of the train images (≥ 5
            # images) so the run still has a val curve; ft-1023-1911 trained blind
            "val_images": 0
            if val_jsonl
            else (min(50, max(1, n_images // 10)) if n_images >= 5 else 0),
            **({"val": [str(p) for p in val_jsonl]} if val_jsonl else {}),
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
        self.procs: dict[str, subprocess.Popen] = {}
        self.cancelled: set[str] = set()

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
        tl = paths.RUNS / name / "train.log"
        if st.get("state") == "error" and tl.exists():
            st["log_tail"] = tl.read_text(errors="replace").splitlines()[-20:]
        return st

    def cancel(self, name: str) -> bool:
        """Stops the running subprocess; the last saved checkpoint stays."""
        p = self.procs.get(name)
        if p is None or p.poll() is not None:
            return False
        self.cancelled.add(name)
        p.terminate()
        return True

    def delete(self, name: str) -> None:
        import shutil

        t = self.threads.get(name)
        if t and t.is_alive():
            raise RuntimeError("還在跑，先取消")
        shutil.rmtree(paths.RUNS / name, ignore_errors=True)
        self.threads.pop(name, None)

    def _call(self, name: str, args: list[str], log) -> int:
        """Runs one subprocess, remembering it so ``cancel`` can stop it."""
        p = subprocess.Popen(args, cwd=paths.ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.procs[name] = p
        try:
            return p.wait()
        finally:
            self.procs.pop(name, None)

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
    def start(
        self,
        name: str,
        steps: int,
        lr: float,
        rank: int = DEFAULT_RANK,
        rows: list[dict] | None = None,
        val_manifests: dict[str, Path] | None = None,
        region_head: bool = False,
        datasets: list[str] | None = None,
    ) -> dict:
        """``rows``: the training rows (default: everything reviewed in the workspace);
        ``val_manifests``: extra eval suites (a dataset's own validation split) scored for
        base and adapter alike; ``region_head`` off keeps the adapter Photon-compatible."""
        if any(t.is_alive() for t in self.threads.values()):
            raise RuntimeError("a training pipeline is already running")
        if (paths.RUNS / name).exists():
            raise RuntimeError(f"run {name} exists")
        if rows is None:
            exp = export_train(self.ws, None, name)
            train_path, n_rows = Path(exp["path"]), exp["rows"]
        else:
            train_path = paths.WORKSPACE / "export" / f"{name}.jsonl"
            write_rows(train_path, rows, ensure_ascii=False)
            n_rows = len(rows)
        if n_rows == 0:
            raise RuntimeError("nothing to train on: review some data first")
        n_images = len({json.loads(line)["image"] for line in open(train_path)})
        skills: dict[str, int] = {}
        for line in open(train_path):
            k = json.loads(line)["skill"]
            skills[k] = skills.get(k, 0) + 1
        cfg = write_config(
            name,
            train_path,
            n_images,
            steps,
            lr,
            rank,
            region_head,
            list((val_manifests or {}).values()),
        )
        st = {
            "name": name,
            "state": "running",
            "phase": "training",
            "started": datetime.now().isoformat(timespec="seconds"),
            "steps": steps,
            "lr": lr,
            "rank": rank,
            "rows": n_rows,
            "skills": skills,
            "images": n_images,
            "region_head": region_head,
            "datasets": datasets or [],
            "val_manifests": {k: str(v) for k, v in (val_manifests or {}).items()},
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
            rc = self._call(name, self._cmd("train", str(cfg)), log)
            if name in self.cancelled:
                st["state"], st["phase"] = "cancelled", "cancelled"
                return
            if rc != 0:
                raise RuntimeError(f"training exited with {rc} (see runs/{name}/train.log)")
            adapter = run_dir / f"{name}.pt"
            if not adapter.exists():
                raise RuntimeError("training produced no adapter")
            suites: dict[str, Path] = {
                s: paths.MANIFESTS / f"{s}.jsonl" for s in paths.HELDOUT_SUITES
            }
            suites.update({k: Path(v) for k, v in st.get("val_manifests", {}).items()})
            for suite, manifest in suites.items():
                if not manifest.exists():
                    continue
                st["phase"] = f"eval:{suite}"
                self._write(name, st)
                base_rep = paths.BASELINES / f"base-{manifest.stem}.json"
                if not base_rep.exists():  # a dataset's val split: score base once
                    self._call(
                        name,
                        self._cmd("eval", str(manifest), "--batch", "32", "--out", str(base_rep)),
                        log,
                    )
                rc = self._call(
                    name,
                    self._cmd("eval", str(manifest), "--adapter", str(adapter), "--batch", "32"),
                    log,
                )
                if name in self.cancelled:
                    st["state"], st["phase"] = "cancelled", "cancelled"
                    return
                rep = run_dir / f"eval-{name}-{manifest.stem}.json"
                if rc == 0 and rep.exists():
                    m = json.loads(rep.read_text())["metrics"]
                    st["evals"][suite] = {
                        k.split("/")[0]: v for k, v in m.items() if k.endswith("/macro")
                    }
                    self._write(name, st)
            # the gate is the v1 criterion (GOALS §1): the target suite only; the other
            # held-out suites are shown as references, not averaged in
            held = [s for s in st["evals"] if s == paths.HELDOUT_SUITES[0]]
            base_reports = [paths.BASELINES / f"base-{s}.json" for s in held]
            base_reports = [p for p in base_reports if p.exists()]
            if base_reports:
                adapter_reports = [run_dir / f"eval-{name}-{s}.json" for s in held]
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
