"""Where things live on this machine (one place for every tool; GOALS §7, §9).

Everything under the repo is gitignored: ``model-cache/`` (upstream weights + tokenizer),
``runs/`` (training runs, baselines, eval reports), ``adapters/`` (packaged adapters),
``workspace/`` (the UI's SQLite + images), ``data/`` (manifest cache). Data Core release
data is read in place on the data host; held-out images never leave it.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = Path(os.environ.get("VLMFORGE_MODEL_CACHE", ROOT / "model-cache" / "moondream31"))
WEIGHTS = MODEL_CACHE / "weights" / "model.safetensors"
TOKENIZER = MODEL_CACHE / "tokenizer"

# the judge (GOALS §3): ``judge`` → Qwen3-VL-32B-Instruct (a symlink), ``judge-8b`` →
# Qwen3-VL-8B-Instruct, the fallback when the shared GPU cannot fit the 32B. The GB figures
# are what each needs free before loading (weights + batch-of-8 activations).
JUDGE_MODEL = Path(os.environ.get("VLMFORGE_JUDGE_MODEL", ROOT / "model-cache" / "judge"))
JUDGE_MODEL_SMALL = Path(os.environ.get("VLMFORGE_JUDGE_SMALL", ROOT / "model-cache" / "judge-8b"))
JUDGE_GB, JUDGE_SMALL_GB = 70.0, 22.0

RUNS = ROOT / "runs"
BASELINES = RUNS / "baselines"
ADAPTERS = ROOT / "adapters"
WORKSPACE = ROOT / "workspace"
MANIFESTS = ROOT / "data" / "manifests"

DATA_CORE_VLM = Path(os.environ.get("VLMFORGE_DATA_CORE", "/home/paul/SyncAI-Data-Core-vlm"))
RELEASES = DATA_CORE_VLM / "datasets" / "vlm-corpus" / "releases"
HELDOUT_RELEASE = RELEASES / "syncvision-heldout-eval"

# the held-out suites (GOALS §4), in the order reports list them
HELDOUT_SUITES = ("heldout-cctv", "heldout-crowdhuman-val", "heldout-people-posture-gold")
