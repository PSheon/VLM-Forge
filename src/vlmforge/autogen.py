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

import json
import re

from .evaluate import match_boxes, match_points_in_boxes, number
from .judge import parse_json

MAX_OBJECTS = 6
MAX_INSTANCES = 12  # per object type; keeps the teacher's JSON inside its token budget
MAX_QUESTIONS = 8

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

Answer with JSON only, exactly this shape:
{{"objects": [{{"name": "<short noun, singular>", "count": <number of instances>}}, ...],
 "questions": [{{"q": "<question>", "a": "<short answer>"}}, ...],
 "caption": "<one or two factual sentences>"}}

Rules:
- objects: up to {max_objects} object types that are clearly visible, each with how many \
instances are visible. Prefer people, vehicles, forklifts, pallets, boxes, shelves, doors, \
safety equipment, machinery, furniture.
- questions, part 1 (3 to 4): a count ("How many ... are there?" answered with a digit), a \
yes/no question, a colour or attribute, a location or spatial relation, an activity.
- questions, part 2 (2 to 3 inspection questions, yes/no or a digit, only ones this image can \
answer): {topics}.
- answers at most 5 words; a question the image cannot settle must not be asked.
- caption: only what is clearly visible, no guesses.
{focus}"""


GROUND_PROMPT = """Locate every "{name}" in this image (about {count}). Answer with JSON only: a \
list of boxes, one per instance, as integers 0-1000 (fraction of the image width / height \
times 1000), tight around the object:
[[x0, y0, x1, y1], [x0, y0, x1, y1], ...]
At most {max_instances} boxes. If there is none, answer []."""


def teacher_prompt(focus: str = "", topics: list[str] | None = None) -> str:
    extra = f"- Focus on: {focus}" if focus else ""
    tl = "; ".join(INSPECTION_TOPICS + [t for t in (topics or []) if t.strip()])
    return TEACHER_PROMPT.format(max_objects=MAX_OBJECTS, focus=extra, topics=tl)


def ground_prompt(name: str, count: int) -> str:
    return GROUND_PROMPT.format(name=name, count=count, max_instances=MAX_INSTANCES)


def parse_boxes(text: str) -> list[list[float]]:
    """A grounding answer -> normalised boxes; duplicates (the model looping) dropped."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    s = m.group(0)
    try:
        raw = json.loads(s)
    except json.JSONDecodeError:
        # cut back to the last complete inner box and close the list
        cut = s.rfind("]")
        try:
            raw = json.loads(s[: cut + 1].rstrip().rstrip(",") + "]")
        except json.JSONDecodeError:
            raw = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", s)
    if raw and all(isinstance(x, int | float) for x in raw) and len(raw) == 4:
        raw = [raw]
    out: list[list[float]] = []
    for b in raw or []:
        if not isinstance(b, list | tuple) or len(b) != 4:
            continue
        try:
            v = [min(1.0, max(0.0, float(x) / 1000.0)) for x in b]
        except (TypeError, ValueError):
            continue
        v = [round(x, 4) for x in v]
        if v[2] > v[0] and v[3] > v[1] and v not in out:
            out.append(v)
        if len(out) >= MAX_INSTANCES:
            break
    return out


_BOX_GROUPS = re.compile(r'("boxes"\s*:\s*)((?:\[[^\[\]]*\]\s*,?\s*)+)')


def repair_json(text: str) -> str:
    """Two things the 32B teacher gets wrong: ``"boxes": [1,2,3,4]`` or ``[..], [..]`` without
    the outer list (wrapped here), and output cut off by the token budget (closed here: the
    text is trimmed back to the last complete ``}`` and the open brackets are closed)."""
    m = re.search(r"\{.*", text, re.S)
    if not m:
        return text
    s = m.group(0)

    def wrap(mm: re.Match) -> str:
        groups = mm.group(2).rstrip().rstrip(",")
        return f"{mm.group(1)}[{groups}]"

    s = _BOX_GROUPS.sub(wrap, s)
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    cut = s.rfind("}")
    s = s[: cut + 1] if cut > 0 else s
    depth_sq = s.count("[") - s.count("]")
    depth_br = s.count("{") - s.count("}")
    # close the innermost open structures in an order that keeps lists inside objects valid
    tail = ""
    stack = []
    for ch in s:
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}" and stack:
            stack.pop()
    for ch in reversed(stack):
        tail += "]" if ch == "[" else "}"
    s = s.rstrip().rstrip(",") + tail
    return s if (depth_sq or depth_br) else s


def parse_teacher(text: str) -> dict:
    """First teacher call -> {"objects": [{"name", "count", "boxes": []}], "questions": [...],
    "caption": str}; boxes are filled by the grounding calls. Malformed parts are dropped."""
    raw = parse_json(repair_json(text))
    objects = []
    seen: set[str] = set()
    for o in raw.get("objects", []) if isinstance(raw.get("objects"), list) else []:
        if not isinstance(o, dict) or not str(o.get("name", "")).strip():
            continue
        name = str(o["name"]).strip().lower()[:40]
        if name in seen:
            continue
        seen.add(name)
        try:
            count = max(0, int(o.get("count", 0)))
        except (TypeError, ValueError):
            count = 0
        boxes = []
        bl = o.get("boxes", []) or []  # tolerated if the model adds them anyway
        if len(bl) == 4 and all(isinstance(x, int | float) for x in bl):
            bl = [bl]
        for b in bl[:MAX_INSTANCES]:
            if isinstance(b, list | tuple) and len(b) == 4:
                try:
                    v = [min(1.0, max(0.0, float(x) / 1000.0)) for x in b]
                except (TypeError, ValueError):
                    continue
                if v[2] > v[0] and v[3] > v[1]:
                    boxes.append([round(x, 4) for x in v])
        objects.append({"name": name, "count": count or len(boxes), "boxes": boxes})
    questions = []
    for q in raw.get("questions", []) if isinstance(raw.get("questions"), list) else []:
        if isinstance(q, dict) and str(q.get("q", "")).strip() and str(q.get("a", "")).strip():
            questions.append({"q": str(q["q"]).strip()[:200], "a": str(q["a"]).strip()[:80]})
    caption = str(raw.get("caption", "")).strip()[:600]
    return {
        "objects": objects[:MAX_OBJECTS],
        "questions": questions[:MAX_QUESTIONS],
        "caption": caption,
    }


def items_from_teacher(t: dict, skills: set[str]) -> list[dict]:
    """(skill, prompt, teacher) tuples for one image."""
    out = []
    for o in t["objects"]:
        if o.get("skipped"):  # more instances than the teacher can box reliably
            continue
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
        return {"verdict": "fix", "note": "模型沒作答", "model": "agreement"}
    if skill == "detect":
        tp, fp, fn = match_boxes(pred.get("boxes") or [], teacher.get("boxes") or [])
        ok = fp == 0 and fn == 0
        note = f"模型 {tp + fp} 框 vs 預標 {tp + fn} 框：{tp} 對上、{fp} 多出、{fn} 漏掉"
        return {"verdict": "accept" if ok else "fix", "note": note, "model": "agreement"}
    if skill == "point":
        tp, fp, fn = match_points_in_boxes(pred.get("points") or [], teacher.get("boxes") or [])
        ok = fp == 0 and fn == 0
        note = f"模型 {tp + fp} 點 vs 預標 {tp + fn} 框：{fp} 點落在框外、{fn} 框沒有點"
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
            "note": f"模型「{s[:60]}」vs 預標「{t[:60]}」",
            "answer": t,
            "model": "agreement",
        }
    return {"verdict": "fix", "note": "caption 交給 judge 查", "model": "agreement"}


def auto_label(skill: str, pred: dict | None, teacher: dict) -> dict:
    """The label an accepted auto item exports: teacher text, student geometry."""
    if skill in ("query", "caption"):
        return {"text": teacher.get("text", "")}
    if skill == "detect":
        return {"boxes": (pred or {}).get("boxes") or teacher.get("boxes") or []}
    return {"points": (pred or {}).get("points") or teacher.get("points") or []}
