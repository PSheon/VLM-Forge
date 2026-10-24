"""``forge`` command line.

forge serve                     the UI (everything below is also a button there)
forge manifests                 held-out manifests from the Data Core release (GOALS §7)
forge eval SUITE [--adapter P]  score base (or base + adapter) on a held-out suite
forge train CONFIG              LoRA fine-tune (vlmforge.train); the UI writes
                                runs/<name>/config.yaml, so a run can be repeated by hand
forge gate --base .. --adapter .. --skill multi   the pass / fail rule (GOALS §6)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=f"forge {__version__}")
    sub = p.add_subparsers(dest="command")

    m = sub.add_parser("manifests", help="build the held-out manifests from the release")
    m.add_argument("--release", type=Path, default=paths.HELDOUT_RELEASE)
    m.add_argument("--out", type=Path, default=paths.MANIFESTS)

    e = sub.add_parser("eval", help="score a model on a suite (a manifest name or path)")
    e.add_argument("suite")
    e.add_argument("--adapter", type=Path, help="adapter .pt; omitted = base model")
    e.add_argument("--out", type=Path, help="report path (default: baselines/ or next to adapter)")
    e.add_argument("--batch", type=int, default=64)

    t = sub.add_parser("train", help="LoRA fine-tune from a YAML config")
    t.add_argument("config")

    g = sub.add_parser("gate", help="compare base vs adapter reports (GOALS §6)")
    g.add_argument("--base", type=Path, nargs="+", required=True)
    g.add_argument("--adapter", type=Path, nargs="+", required=True)
    g.add_argument("--skill", required=True, help="detect | point | query | caption | multi")
    g.add_argument("--json", type=Path, help="also write the result here")

    s = sub.add_parser("serve", help="the UI (GOALS §8)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8796)
    s.add_argument("--batch", type=int, default=16)

    return p


def _suite_path(suite: str) -> Path:
    p = Path(suite)
    return p if p.exists() else paths.MANIFESTS / f"{suite}.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "manifests":
        from .data import build_heldout

        for name, n in build_heldout(args.release, args.out).items():
            print(f"{name} {n}")
        return 0
    if args.command == "eval":
        from .evaluate import evaluate, macro

        manifest = _suite_path(args.suite)
        out = args.out
        if out is None:
            out = (
                paths.BASELINES / f"base-{manifest.stem}.json"
                if args.adapter is None
                else args.adapter.parent / f"eval-{args.adapter.stem}-{manifest.stem}.json"
            )
        res = evaluate(manifest, out, args.adapter, args.batch)
        print(f"wrote {out}")
        print({k: round(v, 4) for k, v in macro(res["metrics"]).items()})
        return 0
    if args.command == "train":
        from .train import main as train_main

        sys.argv = ["forge train", args.config]
        train_main()
        return 0
    if args.command == "gate":
        import json

        from .gate import combined_macro, format_gate, gate

        res = gate(combined_macro(args.base), combined_macro(args.adapter), args.skill)
        print(format_gate(res))
        if args.json:
            args.json.write_text(json.dumps(res, indent=1))
        return 0 if res["passed"] else 1
    if args.command == "serve":
        import os

        import uvicorn

        # shared GPU: let the allocator grow in segments instead of fragmenting to OOM
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        from .server.api import create_app
        from .server.engine import Engine
        from .server.workspace import Workspace

        app = create_app(
            Workspace(paths.WORKSPACE), Engine(batch_size=args.batch), paths.ROOT / "web" / "dist"
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
