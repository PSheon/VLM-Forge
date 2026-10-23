"""The judge: a larger local VLM checks every prediction before a human sees it (GOALS §6).

Flywheel: Moondream predicts -> the judge accepts or flags -> humans review only the flagged
items plus a spot-check sample of the accepted ones. Default judge: Qwen3-VL-32B-Instruct
(Apache-2.0), run locally through transformers, so images never leave the host. Same model
VisionTraining uses for its caption spot checks.

Verdict per item: ``{"verdict": "accept" | "fix", "note": str, ...}``; ``fix`` means a human
must look (wrong / missing boxes or points, a wrong answer, unsupported caption phrases).
The judge never writes labels: an accepted prediction is exported as ``machine_reviewed``,
a human's as ``human`` (Data Core rules §6-§7).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MAX_SIDE = 1280  # judge input size; CCTV frames are 2560 px wide

DETECT_PROMPT = """The image has {n} red boxes numbered 1 to {n}. Each box is supposed to \
contain exactly one "{obj}" (and no two boxes the same one).

Check every box and look for "{obj}" instances that have no box. Answer with JSON only:
{{"wrong": [numbers of boxes that are not a {obj}, badly placed, or duplicate], \
"missed": <number of clearly visible {obj} instances without a box>, "note": "<one line>"}}"""

DETECT_EMPTY_PROMPT = """The model says there is no "{obj}" in this image.

Is that right? Answer with JSON only:
{{"missed": <number of clearly visible {obj} instances>, "note": "<one line>"}}"""

POINT_PROMPT = """The image has {n} red dots numbered 1 to {n}. Each dot is supposed to sit \
on one "{obj}" (one dot per instance).

Answer with JSON only:
{{"wrong": [numbers of dots that are not on a {obj} or duplicate one], \
"missed": <number of clearly visible {obj} instances without a dot>, "note": "<one line>"}}"""

QUERY_PROMPT = """Question about this image: {q}
Proposed answer: {a}

Is the proposed answer correct? Be strict about counts and yes/no. Answer with JSON only:
{{"correct": true or false, "answer": "<the correct short answer>", "note": "<one line>"}}"""

CAPTION_PROMPT = """Caption written for this image:

{text}

List every phrase the image does NOT support: objects that are not there, wrong counts, \
wrong colours or attributes, wrong actions, text or brands that cannot be read, wrong \
spatial relations. Do not flag vague wording or omissions. Quote phrases exactly. Answer \
with JSON only:
{{"unsupported": ["phrase", ...], "note": "<one line>"}}"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) else {}


def render(image: Image.Image, pred: dict) -> Image.Image:
    """Boxes / points drawn on a copy, numbered from 1, red with a white outline."""
    im = image.convert("RGB")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, W // 500)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, W // 60))
    except OSError:
        font = ImageFont.load_default()

    def tag(x, y, k):
        t = str(k)
        bb = d.textbbox((x, y), t, font=font)
        d.rectangle((bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2), fill=(220, 0, 0))
        d.text((x, y), t, fill="white", font=font)

    for k, b in enumerate(pred.get("boxes") or [], 1):
        x0, y0, x1, y1 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        d.rectangle((x0, y0, x1, y1), outline="white", width=lw + 2)
        d.rectangle((x0, y0, x1, y1), outline=(220, 0, 0), width=lw)
        tag(x0 + lw, y0 + lw, k)
    r = max(4, W // 200)
    for k, p in enumerate(pred.get("points") or [], 1):
        x, y = p[0] * W, p[1] * H
        d.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), fill="white")
        d.ellipse((x - r, y - r, x + r, y + r), fill=(220, 0, 0))
        tag(x + r + 2, y - r, k)
    return im


def prompt_for(skill: str, prompt: str, pred: dict) -> str:
    if skill == "detect":
        n = len(pred.get("boxes") or [])
        return (
            DETECT_PROMPT.format(n=n, obj=prompt) if n else DETECT_EMPTY_PROMPT.format(obj=prompt)
        )
    if skill == "point":
        n = len(pred.get("points") or [])
        return POINT_PROMPT.format(n=n, obj=prompt) if n else DETECT_EMPTY_PROMPT.format(obj=prompt)
    if skill == "query":
        return QUERY_PROMPT.format(q=prompt, a=pred.get("text", ""))
    return CAPTION_PROMPT.format(text=pred.get("text", ""))


def verdict_from(skill: str, raw: dict) -> dict:
    """Judge JSON -> ``{"verdict", "note", ...}``. Unparseable output is a ``fix`` (a human
    looks), never an accept."""
    note = str(raw.get("note", ""))[:200]
    if not raw:
        return {"verdict": "fix", "note": "judge gave no parseable answer", "raw_empty": True}
    if skill in ("detect", "point"):
        wrong = [int(x) for x in raw.get("wrong", []) if str(x).lstrip("-").isdigit()]
        missed = raw.get("missed", 0)
        missed = int(missed) if str(missed).lstrip("-").isdigit() else 1
        ok = not wrong and missed <= 0
        return {
            "verdict": "accept" if ok else "fix",
            "wrong": wrong,
            "missed": missed,
            "note": note,
        }
    if skill == "query":
        ok = raw.get("correct") is True
        return {
            "verdict": "accept" if ok else "fix",
            "answer": str(raw.get("answer", "")),
            "note": note,
        }
    uns = [str(p) for p in raw.get("unsupported", []) if isinstance(p, str)]
    return {"verdict": "accept" if not uns else "fix", "unsupported": uns, "note": note}


class Judge:
    """Qwen3-VL through transformers; one image per generate (the judge is not the bottleneck,
    the human is)."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        name: str | None = None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_dir = str(model_dir)
        self.name = name or Path(self.model_dir).resolve().name  # recorded on every verdict
        self.proc = AutoProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir, dtype=torch.bfloat16
        ).to(device)
        self.model.eval()
        self.device, self.max_new_tokens = device, max_new_tokens

    def ask(self, image: Image.Image, text: str) -> str:
        import torch

        msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
            }
        ]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1] :]
        return self.proc.decode(new, skip_special_tokens=True)

    def ask_batch(self, pairs: list[tuple[Image.Image, str]]) -> list[str]:
        """Several (image, prompt) pairs in one generate (left padding); the teacher's
        throughput, not its latency, is what a 300-image batch needs."""
        import torch

        if not pairs:
            return []
        msgs = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": im}, {"type": "text", "text": text}],
                }
            ]
            for im, text in pairs
        ]
        self.proc.tokenizer.padding_side = "left"
        inputs = self.proc.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        n = inputs["input_ids"].shape[1]
        return [self.proc.decode(row[n:], skip_special_tokens=True) for row in out]

    def generate(self, image_path: str, focus: str = "", topics: list[str] | None = None) -> dict:
        return self.generate_batch([image_path], focus, topics)[0]

    def fixed_batch(self, image_paths: list[str], fixed: dict) -> list[dict]:
        """Teacher answers to a dataset's fixed prompts (``vlmforge.autogen.parse_fixed``),
        then the same grounding calls as ``generate_batch``."""
        from .autogen import fixed_prompt, parse_teacher

        images = [render(Image.open(p), {}) for p in image_paths]
        raws = self.ask_batch([(im, fixed_prompt(fixed)) for im in images])
        out = [parse_teacher(r) for r in raws]
        for t, r in zip(out, raws, strict=True):
            t["raw"] = r[:2000]
            t["model"] = self.name
        self._ground(images, out)
        return out

    def generate_batch(
        self, image_paths: list[str], focus: str = "", topics: list[str] | None = None
    ) -> list[dict]:
        """Teacher mode (``vlmforge.autogen``) for several images: one batched call for object
        names + counts, questions and caption, then the grounding prompts of every image
        batched together (long box lists made the 32B model loop, so one short call per
        object type). Types with more than MAX_INSTANCES instances are not boxed."""
        from .autogen import parse_teacher, teacher_prompt

        images = [render(Image.open(p), {}) for p in image_paths]
        raws = self.ask_batch([(im, teacher_prompt(focus, topics)) for im in images])
        out = []
        for raw_text in raws:
            t = parse_teacher(raw_text)
            t["raw"] = raw_text[:2000]
            t["model"] = self.name
            out.append(t)
        self._ground(images, out)
        return out

    def _ground(self, images: list, teachers: list[dict]) -> None:
        """One short grounding call per (image, object type), batched 8 at a time; types with
        more than MAX_INSTANCES instances are marked ``skipped`` instead."""
        from .autogen import MAX_INSTANCES, ground_prompt, parse_boxes

        asks: list[tuple[int, dict]] = []
        for i, t in enumerate(teachers):
            for o in t["objects"]:
                if o["count"] > MAX_INSTANCES:
                    o["skipped"] = True
                elif o["count"] > 0 and not o["boxes"]:
                    asks.append((i, o))
        for k in range(0, len(asks), 8):
            chunk = asks[k : k + 8]
            answers = self.ask_batch(
                [(images[i], ground_prompt(o["name"], o["count"])) for i, o in chunk]
            )
            for (_, o), a in zip(chunk, answers, strict=True):
                o["boxes"] = parse_boxes(a)

    def judge(self, image_path: str, skill: str, prompt: str, pred: dict) -> dict:
        image = Image.open(image_path)
        shown = render(image, pred) if skill in ("detect", "point") else render(image, {})
        raw_text = self.ask(shown, prompt_for(skill, prompt, pred))
        v = verdict_from(skill, parse_json(raw_text))
        v["model"] = self.name
        v["raw"] = raw_text[:500]
        return v
