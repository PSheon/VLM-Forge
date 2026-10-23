"""The stage gate (GOALS §6): an adapter's target skill scores >= base on held-out, and no
other skill drops by more than ``tolerance`` (1 point) macro. Scores come from the eval
reports ``forge eval`` writes; a skill missing from both reports (no gold yet) is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import macro

TOLERANCE = 0.01


def combined_macro(reports: list[Path]) -> dict[str, float]:
    """Per-skill macro over several suites: the mean of each suite's ``<skill>/macro``."""
    acc: dict[str, list[float]] = {}
    for p in reports:
        for skill, v in macro(json.loads(Path(p).read_text())["metrics"]).items():
            acc.setdefault(skill, []).append(v)
    return {s: sum(v) / len(v) for s, v in acc.items()}


def gate(
    base: dict[str, float], adapter: dict[str, float], target: str, tolerance: float = TOLERANCE
) -> dict:
    """``target``: the fine-tuned skill, or ``multi`` (every skill must hold >= base - tol
    and at least one must improve)."""
    skills = sorted(set(base) & set(adapter))
    if target != "multi" and target not in skills:
        raise ValueError(f"no {target} scores in both reports (skills: {skills})")
    rows = []
    for s in skills:
        delta = adapter[s] - base[s]
        if target == "multi":
            ok = delta >= -tolerance
        else:
            ok = delta >= 0 if s == target else delta >= -tolerance
        rows.append({"skill": s, "base": base[s], "adapter": adapter[s], "delta": delta, "ok": ok})
    passed = all(r["ok"] for r in rows)
    if target == "multi":
        passed = passed and any(r["delta"] > 0 for r in rows)
    return {"target": target, "tolerance": tolerance, "passed": passed, "skills": rows}


def format_gate(result: dict) -> str:
    lines = [f"gate {result['target']}: {'PASS' if result['passed'] else 'FAIL'}"]
    for r in result["skills"]:
        mark = "ok " if r["ok"] else "BAD"
        lines.append(
            f"  {mark} {r['skill']:<8} base {r['base']:.4f}  adapter {r['adapter']:.4f}  "
            f"delta {r['delta']:+.4f}"
        )
    return "\n".join(lines)
