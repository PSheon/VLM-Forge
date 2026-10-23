"""Auto-generated training data (GOALS §6): the tool picks images, writes the questions and
the teacher answers, the student answers, the judge compares, a human checks per image.

Teacher = the judge model (Qwen3-VL-32B). One call per image returns objects with boxes,
3-5 questions with short answers and a caption; they become detect / point / query / caption
items whose ``teacher`` field holds the teacher's answer. The student (Moondream) predicts
``pred``; ``agree`` compares the two without another model call (boxes by IoU matching,
points inside the teacher's boxes, answers by number / yes-no / text); captions are judged
for unsupported phrases. Disagreement = ``fix`` = a human decides.

Labels for accepted items: text comes from the teacher, boxes / points from the student
(its box style is what the region head learns). A human's edit always wins.
"""

from __future__ import annotations

import re

from .evaluate import match_boxes, match_points_in_boxes, number
from .judge import parse_json

MAX_OBJECTS = 6

# Industrial inspection questions every image gets a few of (Paul, 2026-10-23); a queue can
# add its own topics on top.
INSPECTION_TOPICS = [
    "whether any staff / worker is present on duty",
    "whether each person wears a safety helmet, a high-visibility vest, a mask, gloves",
    "whether an exit, aisle or fire extinguisher is blocked",
    "whether a door or gate is open",
    "whether there is liquid, debris or an obstacle on the floor",
    "whether a forklift or machine is running or moving",
    "whether the lights are on",
    "whether anyone is in a restricted or dangerous area",
    "whether an object is left unattended",
]

TEACHER_PROMPT = """Look at this image and write training data for a small vision-language \
model used in factories, warehouses and on robots.

Answer with JSON only:
{{"objects": [{{"name": "<short noun, singular>", "boxes": [[x0, y0, x1, y1], ...]}}, ...],
 "questions": [{{"q": "<question>", "a": "<short answer>"}}, ...],
 "caption": "<one or two factual sentences>"}}

Rules:
- objects: up to {max_objects} object types that are clearly visible; one box per instance, \
tight, as integers 0-1000 (fraction of the image width / height times 1000). Prefer people, \
vehicles, forklifts, pallets, boxes, shelves, doors, safety equipment, machinery, furniture.
- questions, part 1 (3 to 4): a count ("How many ... are there?" answered with a digit), a \
yes/no question, a colour or attribute, a location or spatial relation, an activity.
- questions, part 2 (2 to 3 inspection questions, yes/no or a digit, only ones this image can \
answer): {topics}.
- answers at most 5 words; a question the image cannot settle must not be asked.
- caption: only what is clearly visible, no guesses.
{focus}"""


def teacher_prompt(focus: str = "", topics: list[str] | None = None) -> str:
    extra = f"- Focus on: {focus}" if focus else ""
    tl = "; ".join(INSPECTION_TOPICS + [t for t in (topics or []) if t.strip()])
    return TEACHER_PROMPT.format(max_objects=MAX_OBJECTS, focus=extra, topics=tl)


def parse_teacher(text: str) -> dict:
    """Judge output -> {"objects": [{"name", "boxes": [[0-1 floats]]}], "questions": [...],
    "caption": str}; malformed parts are dropped, never guessed."""
    raw = parse_json(text)
    objects = []
    for o in raw.get("objects", []) if isinstance(raw.get("objects"), list) else []:
        if not isinstance(o, dict) or not str(o.get("name", "")).strip():
            continue
        boxes = []
        for b in o.get("boxes", []) or []:
            if isinstance(b, list | tuple) and len(b) == 4:
                try:
                    v = [min(1.0, max(0.0, float(x) / 1000.0)) for x in b]
                except (TypeError, ValueError):
                    continue
                if v[2] > v[0] and v[3] > v[1]:
                    boxes.append([round(x, 4) for x in v])
        objects.append({"name": str(o["name"]).strip().lower()[:40], "boxes": boxes})
    questions = []
    for q in raw.get("questions", []) if isinstance(raw.get("questions"), list) else []:
        if isinstance(q, dict) and str(q.get("q", "")).strip() and str(q.get("a", "")).strip():
            questions.append({"q": str(q["q"]).strip()[:200], "a": str(q["a"]).strip()[:80]})
    caption = str(raw.get("caption", "")).strip()[:600]
    return {"objects": objects[:MAX_OBJECTS], "questions": questions[:5], "caption": caption}


def items_from_teacher(t: dict, skills: set[str]) -> list[dict]:
    """(skill, prompt, teacher) tuples for one image."""
    out = []
    for o in t["objects"]:
        if "detect" in skills:
            out.append({"skill": "detect", "prompt": o["name"], "teacher": {"boxes": o["boxes"]}})
        if "point" in skills and o["boxes"]:
            pts = [[round((b[0] + b[2]) / 2, 4), round((b[1] + b[3]) / 2, 4)] for b in o["boxes"]]
            out.append(
                {
                    "skill": "point",
                    "prompt": o["name"],
                    "teacher": {"points": pts, "boxes": o["boxes"]},
                }
            )
    if "query" in skills:
        for q in t["questions"]:
            out.append({"skill": "query", "prompt": q["q"], "teacher": {"text": q["a"]}})
    if "caption" in skills and t["caption"]:
        out.append({"skill": "caption", "prompt": "normal", "teacher": {"text": t["caption"]}})
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def agree(skill: str, pred: dict | None, teacher: dict) -> dict:
    """Student vs teacher -> a judge-style verdict (``accept`` / ``fix``) with a note."""
    if pred is None:
        return {"verdict": "fix", "note": "student gave no answer", "model": "agreement"}
    if skill == "detect":
        tp, fp, fn = match_boxes(pred.get("boxes") or [], teacher.get("boxes") or [])
        ok = fp == 0 and fn == 0
        note = f"student {tp + fp} boxes vs teacher {tp + fn}: {tp} match, {fp} extra, {fn} missed"
        return {"verdict": "accept" if ok else "fix", "note": note, "model": "agreement"}
    if skill == "point":
        tp, fp, fn = match_points_in_boxes(pred.get("points") or [], teacher.get("boxes") or [])
        ok = fp == 0 and fn == 0
        note = f"student {tp + fp} points vs teacher {tp + fn} boxes: {fp} outside, {fn} missed"
        return {"verdict": "accept" if ok else "fix", "note": note, "model": "agreement"}
    if skill == "query":
        s, t = str(pred.get("text", "")), str(teacher.get("text", ""))
        tn = number(t)
        if _norm(t).isdigit() or (tn is not None and len(_norm(t).split()) <= 2):
            ok = number(s) == tn
        elif _norm(t) in ("yes", "no"):
            first = re.findall(r"[a-z]+", s.lower())[:1]
            ok = first == [_norm(t)]
        else:
            ok = _norm(s) == _norm(t) or (_norm(t) in _norm(s) and len(_norm(t)) >= 3)
        return {
            "verdict": "accept" if ok else "fix",
            "note": f'student "{s[:60]}" vs teacher "{t[:60]}"',
            "answer": t,
            "model": "agreement",
        }
    return {"verdict": "fix", "note": "caption needs the judge", "model": "agreement"}


def auto_label(skill: str, pred: dict | None, teacher: dict) -> dict:
    """The label an accepted auto item exports: teacher text, student geometry."""
    if skill in ("query", "caption"):
        return {"text": teacher.get("text", "")}
    if skill == "detect":
        return {"boxes": (pred or {}).get("boxes") or teacher.get("boxes") or []}
    return {"points": (pred or {}).get("points") or teacher.get("points") or []}
