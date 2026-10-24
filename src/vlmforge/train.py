"""LoRA fine-tuning of Moondream 3.1 (GOALS §3, §6).

    numactl --membind=0 uv run forge train runs/<name>/config.yaml   # the UI writes it

What is trained: TextLoRA on the language model (dense MLP layers rank r, MoE experts rank
r/8, engine layout, scale alpha / r folded in at save) and the region decoder head
(``RegionHead``: layer norm + coordinate / size heads). Vision tower, projector, LM head and
the region encoders stay frozen. Master weights are FP32; the forward sees BF16 casts.

Rows (``vlmforge.data``) cover caption / query / detect / point with the model's own prompts
(``prompt_ids``); detect targets are COORD(cx) COORD(cy) SIZE(w, h) per object, point targets
COORD(x) COORD(y), objects sorted left to right. The LM head is supervised on the token
pattern, the region head on the coordinate / size bins.

Self-distilled replay (``replay.ratio``, default 0.5): that share of every batch is the base
model's own greedy answers to generic prompts (questions, captions, detect / point of
objects that are NOT trained). Without it the adapter collapses onto the trained answers.
The pool is generated once and cached next to the run.

Every training row whose image appears in a held-out manifest (``data.exclude``) is dropped.

Outputs in ``runs/<name>/``: ``<name>.pt`` (adapter: ``lora_state_dict``, ``rank``, ``alpha``,
``region_head``), ``ckpt-<step>.pt`` at every save (``evaluate.pick_checkpoint`` keeps the best
as ``<name>.pt``), ``<name>.json`` (metadata), ``log.jsonl``, ``replay.json`` and, while
running, ``resume.pt`` (a restart with the same config resumes from it).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml

from . import paths
from .data import image_key, mixture, split_rows
from .model.moondream import (
    COORD_ID,
    EOS,
    IMAGE_TOKENS,
    SIZE_ID,
    SKILL_STATE,
    Moondream,
    Request,
    prompt_ids,
)
from .model.text import D, RegionHead, RegionModel, TextLoRA
from .model.vision import load_rgb

REPLAY_QUESTIONS = [
    "What is happening in this image?",
    "What is the main object in the image?",
    "What colors are prominent in this image?",
    "Where was this photo taken?",
    "Is there a person in this image? Answer yes or no.",
]
REPLAY_CAPTIONS = ["normal"]
REPLAY_OBJECTS = ["window", "tree", "door", "wheel", "sign", "lamp", "shoe", "hand", "building"]
LORA_PARTS = ("up_a", "up_b", "down_a", "down_b")
DEFAULT_GPU_MEM_GB = 60  # GOALS §3: Forge training stays under 60 GB on the shared GB300

# An answer is a list of (token id, value): value None for a word token, x / y for COORD,
# (w, h) for SIZE. It ends with EOS unless it was truncated.
Answer = list[tuple[int, object]]


def gt_answer(tokenizer, row: dict, max_objects: int, max_text: int) -> tuple[list[int], Answer]:
    """Training row -> (prompt ids, answer). Objects in a fixed left-to-right order; beyond
    ``max_objects`` (the engine's detect limit) the tail is cut and no EOS is supervised."""
    skill = row["skill"]
    item = dict(row)
    if skill == "query":
        item["question"] = row["question"].strip()
    prompt = prompt_ids(tokenizer, skill, item)
    if skill in ("query", "caption"):
        ids = tokenizer.encode(row["answer"].strip()).ids
        ans: Answer = [(t, None) for t in ids[:max_text]]
        return prompt, ans + ([(EOS, None)] if len(ids) <= max_text else [])
    if skill == "detect":
        objs = [
            (
                (b[0] + b[2]) / 2,
                (b[1] + b[3]) / 2,
                max(b[2] - b[0], 1e-3),
                max(b[3] - b[1], 1e-3),
            )
            for b in row["boxes"]
        ]
    elif skill == "point":
        objs = [(p[0], p[1]) for p in row["points"]]
    else:
        raise ValueError(f"unknown skill {skill!r}")
    objs.sort(key=lambda o: (round(o[0], 2), o[1]))
    ans = []
    for o in objs[:max_objects]:
        ans += [(COORD_ID, float(o[0])), (COORD_ID, float(o[1]))]
        if skill == "detect":
            ans.append((SIZE_ID, (float(o[2]), float(o[3]))))
    return prompt, ans + ([(EOS, None)] if len(objs) <= max_objects else [])


def generated_answer(tokens: list[int], values: list) -> Answer:
    """The base model's own output (``Generation``) as an answer, values exactly as decoded."""
    out: Answer = []
    for t, (a, b) in zip(tokens, values, strict=True):
        out.append((t, a if t == COORD_ID else (a, b) if t == SIZE_ID else None))
    return out


def layout(prompt_len: int, answer: Answer):
    """Where an (image, prompt, answer) sample's targets sit in its input sequence.

    The sequence is BOS, 729 image tokens, prompt[1:], answer[:-1]; answer[k] is predicted
    from position ``729 + prompt_len - 1 + k``. Returns (length, token targets [(pos, id)],
    coord targets [(pos, bin)], size targets [(pos, (w bin, h bin))])."""
    start = IMAGE_TOKENS + prompt_len - 1
    tok, coord, size = [], [], []
    for k, (t, v) in enumerate(answer):
        p = start + k
        tok.append((p, t))
        if t == COORD_ID:
            coord.append((p, int(RegionModel.coord_bin(torch.tensor(v)))))
        elif t == SIZE_ID:
            wh = RegionModel.size_bin(torch.tensor(v, dtype=torch.float32))
            size.append((p, (int(wh[0]), int(wh[1]))))
    return start + len(answer), tok, coord, size


def bf16_view(lora: TextLoRA) -> SimpleNamespace:
    """BF16 casts of the FP32 master LoRA for the forward, with the alpha scale on the B
    matrices (gradients flow to the masters)."""

    def cast(m):
        out = {}
        for n in LORA_PARTS:
            t = getattr(m, n)
            out[n] = (t * lora.scale).bfloat16() if n.endswith("_b") else t.bfloat16()
        return SimpleNamespace(**out)

    return SimpleNamespace(dense=[cast(m) for m in lora.dense], moe=[cast(m) for m in lora.moe])


def save_adapter(lora: TextLoRA, head: RegionHead | None, path: Path) -> None:
    sd = {k: v.bfloat16() for k, v in lora.engine_state_dict().items()}
    ckpt = {"lora_state_dict": sd, "rank": lora.rank, "alpha": lora.alpha}
    if head is not None:
        ckpt["region_head"] = {k: v.cpu() for k, v in head.state_dict_upstream().items()}
    torch.save(ckpt, path)


class Batcher:
    """Builds teacher-forced batches: image tokens from the frozen encoder, inputs right-padded
    (padding sits after every real position, so the causal part never sees it)."""

    def __init__(self, m: Moondream, head: RegionHead | None, checkpoint: bool = True):
        self.m, self.head, self.checkpoint = m, head, checkpoint
        self.dev = m.device

    @torch.no_grad()
    def inputs(self, feats: torch.Tensor, samples: list[tuple[list[int], Answer]]):
        text, region, dev = self.m.text, self.m.region, self.dev
        lays = [layout(len(p), a) for p, a in samples]
        T = max(lay[0] for lay in lays)
        x = torch.zeros(len(samples), T, D, device=dev, dtype=text.dtype)
        cb, cp, cv, sb, sp, sv = [], [], [], [], [], []
        for b, (prompt, ans) in enumerate(samples):
            n_p = len(prompt)
            ids = torch.tensor(prompt + [t for t, _ in ans[:-1]], device=dev)
            emb = text.embed(ids)
            x[b, 0] = emb[0]
            x[b, 1 : 1 + IMAGE_TOKENS] = feats[b]
            x[b, 1 + IMAGE_TOKENS : IMAGE_TOKENS + len(ids)] = emb[1:]
            for k, (t, v) in enumerate(ans[:-1]):
                p = IMAGE_TOKENS + n_p + k
                if t == COORD_ID:
                    cb.append(b), cp.append(p), cv.append(v)
                elif t == SIZE_ID:
                    sb.append(b), sp.append(p), sv.append(v)
        if cv:
            x[cb, cp] = region.encode_coord(torch.tensor(cv, device=dev))
        if sv:
            x[sb, sp] = region.encode_size(torch.tensor(sv, device=dev))
        return x, lays

    def loss(self, lora_view, x: torch.Tensor, lays) -> dict[str, torch.Tensor]:
        text, region, dev = self.m.text, self.m.region, self.dev
        hidden = text(x, lora=lora_view, checkpoint=self.checkpoint, moe="grouped")
        L = lambda v, **k: torch.tensor(v, device=dev, **k)  # noqa: E731
        tb, tp, tt, tw, cb, cp, ct, sb, sp, st = ([] for _ in range(10))
        for b, (_, tok, coord, size) in enumerate(lays):
            for p, t in tok:
                tb.append(b), tp.append(p), tt.append(t), tw.append(1.0 / len(tok))
            for p, v in coord:
                cb.append(b), cp.append(p), ct.append(v)
            for p, v in size:
                sb.append(b), sp.append(p), st.append(v)
        ce = F.cross_entropy(text.logits(hidden[tb, tp]).float(), L(tt), reduction="none")
        out = {"tok": (ce * L(tw)).sum() / len(lays)}
        if ct:
            c, _, _ = region.decode(hidden[cb, cp], self.head)
            out["coord"] = F.cross_entropy(c.float(), L(ct))
        if st:
            _, w, h = region.decode(hidden[sb, sp], self.head)
            s = L(st)
            out["size"] = F.cross_entropy(w.float(), s[:, 0]) + F.cross_entropy(h.float(), s[:, 1])
        out["loss"] = sum(out.values())
        return out


def load_rows(cfg: dict) -> tuple[list[dict], dict]:
    """The training mixture minus every image of an excluded (held-out) manifest."""
    d = cfg["data"]
    rows = mixture(d["train"], cfg.get("seed", 0))
    held = set()
    for p in d.get("exclude", []):
        with open(p) as fh:
            held |= {image_key(json.loads(line)["image"]) for line in fh}
    n0 = len(rows)
    rows = [r for r in rows if image_key(r["image"]) not in held]
    stats = {"mixture_rows": n0, "dropped_eval_images": n0 - len(rows), "eval_images": len(held)}
    return rows, stats


def replay_prompts(rcfg: dict, trained_objects: set[str]) -> dict[str, list[tuple[str, dict]]]:
    """Generic prompts for self-distillation, as {"text": [...], "spatial": [...]}. Objects that
    are trained are left out: replaying them would teach the base model's (possibly wrong)
    answer for a trained object."""
    text = [("query", {"question": q}) for q in rcfg.get("questions", REPLAY_QUESTIONS)]
    text += [("caption", {"length": c}) for c in rcfg.get("captions", REPLAY_CAPTIONS)]
    objs = [o for o in rcfg.get("objects", REPLAY_OBJECTS) if o.lower() not in trained_objects]
    spatial = [(s, {"object": o}) for s in ("detect", "point") for o in objs]
    return {"text": text, "spatial": spatial}


def make_replay(m: Moondream, rows: list[dict], rcfg: dict, prompts: dict, seed: int) -> list:
    """Base-model greedy answers on ``replay.images`` training images, each with
    ``per_image_text`` text prompts and ``per_image_spatial`` detect / point prompts."""
    rng = random.Random(seed)
    images = sorted({r["image"] for r in rows})
    images = rng.sample(images, min(rcfg.get("images", 2000), len(images)))
    per = {
        g: min(rcfg.get(f"per_image_{g}", n), len(prompts[g]))
        for g, n in (("text", 2), ("spatial", 1))
    }
    max_text, max_obj = rcfg.get("max_tokens", 256), rcfg.get("max_objects", 50)
    budget = {
        "query": max_text,
        "caption": max_text,
        "detect": 3 * max_obj + 1,
        "point": 2 * max_obj + 1,
    }
    out, t0 = [], time.time()
    chunk = 256
    for i in range(0, len(images), chunk):
        part, ok, arrays = images[i : i + chunk], [], []
        for p in part:
            try:
                arrays.append(load_rgb(p))
                ok.append(p)
            except Exception:  # noqa: BLE001 - a broken image is skipped, not fatal
                continue
        feats = m.encode(arrays)
        jobs, reqs = [], []
        for f, img in zip(feats, ok, strict=True):
            for skill, item in [p for g, n in per.items() for p in rng.sample(prompts[g], n)]:
                jobs.append((img, skill, item))
                reqs.append(
                    Request(
                        f, prompt_ids(m.tokenizer, skill, item), SKILL_STATE[skill], budget[skill]
                    )
                )
        for (img, skill, item), g in zip(jobs, m.generate(reqs), strict=True):
            out.append(
                {
                    "image": img,
                    "skill": skill,
                    "item": item,
                    "tokens": g.tokens,
                    "values": [list(v) for v in g.values],
                }
            )
        print(
            f"replay {len(out)} answers ({i + len(part)}/{len(images)} images, "
            f"{time.time() - t0:.0f}s)",
            flush=True,
        )
    m._decoder = None  # free the generation KV cache before training
    torch.cuda.empty_cache()
    return out


def train(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    name, seed = cfg["name"], cfg.get("seed", 0)
    out = Path(cfg.get("out", paths.RUNS / name))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    gpu_mem = cfg.get("gpu_mem_gb", DEFAULT_GPU_MEM_GB)
    if gpu_mem:  # shared GPU (GOALS §3): OOM here instead of starving SyncVision
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(gpu_mem * 2**30 / total)

    d, o, rcfg = cfg["data"], cfg["optim"], cfg.get("replay", {})
    rows, stats = load_rows(cfg)
    if d.get("val"):  # the datasets' own validation split (server runs): no inner split
        train_rows, val_rows = rows, mixture(d["val"], seed)
    else:
        train_rows, val_rows = split_rows(rows, d.get("val_images", 200), seed)
    print(f"rows train {len(train_rows)} val {len(val_rows)} {stats}", flush=True)
    if not train_rows:
        n_img = len({r["image"] for r in rows})
        raise SystemExit(
            f"no training rows: {len(rows)} rows on {n_img} images, data.val_images="
            f"{d.get('val_images', 200)} held them all out — lower it or add images"
        )
    max_obj, max_text = d.get("max_objects", 50), d.get("max_text", 384)

    m = Moondream(paths.WEIGHTS, paths.TOKENIZER, batch_size=rcfg.get("batch", 64))
    tok = m.tokenizer
    trained = {r["object"].strip().lower() for r in rows if r["skill"] in ("detect", "point")}
    replay: list[dict] = []
    ratio = rcfg.get("ratio", 0.5)
    if ratio > 0:
        prompts = replay_prompts(rcfg, trained)
        pool_path = out / "replay.json"
        key = json.loads(
            json.dumps({"replay": rcfg, "prompts": prompts, "seed": seed, "data": d["train"]})
        )  # as read back from the cache file
        if pool_path.exists() and json.loads(pool_path.read_text())["key"] == key:
            replay = json.loads(pool_path.read_text())["entries"]
            print(f"replay pool reused: {pool_path} ({len(replay)})", flush=True)
        else:
            replay = make_replay(m, train_rows, rcfg, prompts, seed)
            pool_path.write_text(json.dumps({"key": key, "entries": replay}))
        names = [f"{sk}:{next(iter(it.values()))}" for g in prompts.values() for sk, it in g]
        print(f"replay prompts {names}", flush=True)
        if not replay:
            ratio = 0.0
            print("replay pool empty: training without replay", flush=True)

    rank = cfg.get("rank", 16)
    lora = TextLoRA(rank, dev, dtype=torch.float32, alpha=cfg.get("alpha", 2 * rank))
    head = RegionHead(m.region).to(dev) if cfg.get("region_head", True) else None
    params = list(lora.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(
        params, lr=o["lr"], weight_decay=o.get("wd", 0.0), betas=(0.9, 0.95), fused=True
    )
    step, resume = 0, out / "resume.pt"
    if resume.exists():
        st = torch.load(resume, map_location="cpu")
        lora.load_state_dict(st["lora"])
        if head is not None and st.get("head"):
            head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"resumed at step {step} from {resume}", flush=True)
    n_tr = sum(p.numel() for p in params) / 1e6
    print(f"trainable {n_tr:.1f}M (rank {rank}, alpha {lora.alpha}, head {head is not None})")

    def sample(r: dict) -> tuple[str, list[int], Answer]:
        if "tokens" in r:  # replay entry
            return (
                r["image"],
                prompt_ids(tok, r["skill"], r["item"]),
                generated_answer(r["tokens"], r["values"]),
            )
        return (r["image"], *gt_answer(tok, r, max_obj, max_text))

    def plan(s: int, pool: list[dict], b: int, use_replay: bool) -> list[tuple]:
        rng = random.Random(seed * 1_000_003 + s)  # per-step stream: resume is exact
        return [
            sample(
                replay[rng.randrange(len(replay))]
                if use_replay and rng.random() < ratio
                else pool[rng.randrange(len(pool))]
            )
            for _ in range(b)
        ]

    def read(path: str):
        try:
            return load_rgb(path)
        except Exception:  # noqa: BLE001 - a broken image drops the sample
            return None

    io = ThreadPoolExecutor(d.get("workers", 16))
    batcher = Batcher(m, head, cfg.get("gradient_checkpointing", True))

    def fetch(s: int, pool, b: int, use_replay: bool):
        smp = plan(s, pool, b, use_replay)
        return smp, [io.submit(read, im) for im, _, _ in smp]

    wait = [0.0]  # seconds spent waiting for image decoding (data-bound check)

    def run(fetched, grad: bool) -> dict[str, float]:
        smp, futs = fetched
        tw = time.time()
        arrays = [f.result() for f in futs]
        wait[0] += time.time() - tw
        keep = [i for i, a in enumerate(arrays) if a is not None]
        if not keep:
            return {}
        feats = m.vision([arrays[i] for i in keep])
        x, lays = batcher.inputs(feats, [smp[i][1:] for i in keep])
        with torch.set_grad_enabled(grad):
            res = batcher.loss(bf16_view(lora), x, lays)
            if grad:
                res["loss"].backward()
        return {k: float(v) for k, v in res.items()}

    def validate() -> dict[str, float]:
        tot: dict[str, float] = {}
        n = cfg.get("eval_batches", 8)
        for i in range(n):
            res = run(fetch(-1 - i, val_rows, o["batch"], False), grad=False)
            for k, v in res.items():
                tot[k] = tot.get(k, 0.0) + v / n
        return {f"val/{k}": v for k, v in tot.items()}

    log = open(out / "log.jsonl", "a")
    meta_base = {
        "name": name,
        "config": str(cfg_path),
        "data_release": cfg.get("data_release"),
        "rank": rank,
        "alpha": lora.alpha,
        "region_head": head is not None,
        "rows": stats,
        "train_rows": len(train_rows),
        "replay_entries": len(replay),
    }
    accum = o.get("accum", 1)
    if step == 0 and val_rows:
        ev = validate()
        log.write(json.dumps({"step": 0, **ev}) + "\n")
        print(f"eval 0 {ev}", flush=True)
    t0, seen, t_start, n_acc = time.time(), 0, time.time(), 0
    acc: dict[str, float] = {}
    total_micro = o["steps"] * accum
    queue = deque(
        fetch(k, train_rows, o["batch"], True)
        for k in range(step * accum, min(step * accum + d.get("prefetch", 3), total_micro))
    )
    while step < o["steps"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, o)
        for a in range(accum):
            k = step * accum + a + len(queue)
            if k < total_micro:
                queue.append(fetch(k, train_rows, o["batch"], True))
            cur = queue.popleft()
            res = run(cur, grad=True)
            seen += len(cur[0])
            for kk, v in res.items():
                acc[kk] = acc.get(kk, 0.0) + v / accum
        if accum > 1:
            for p in params:
                if p.grad is not None:
                    p.grad.div_(accum)
        gn = torch.nn.utils.clip_grad_norm_(params, o.get("clip", 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        n_acc += 1
        acc["grad_norm"] = acc.get("grad_norm", 0.0) + float(gn)
        if step % cfg.get("log_every", 10) == 0 or step == 1:
            dt = time.time() - t0
            st = {k: v / n_acc for k, v in acc.items()}  # means over the log interval
            st.update(
                lr=lr_at(step, o),
                samples_per_s=seen / dt,
                data_wait=wait[0] / dt,
                mem_gb=torch.cuda.max_memory_allocated() / 2**30,
            )
            log.write(json.dumps({"step": step, **st}) + "\n")
            log.flush()
            print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in st.items()), flush=True)
            t0, seen, n_acc, acc, wait[0] = time.time(), 0, 0, {}, 0.0
        if val_rows and (step % cfg.get("eval_every", 500) == 0 or step == o["steps"]):
            ev = validate()
            log.write(json.dumps({"step": step, **ev}) + "\n")
            log.flush()
            print(f"eval {step} {ev}", flush=True)
        if step % cfg.get("save_every", 500) == 0 or step == o["steps"]:
            save_adapter(lora, head, out / f"{name}.pt")
            save_adapter(lora, head, out / f"ckpt-{step}.pt")  # candidates for pick_checkpoint
            torch.save(
                {
                    "lora": lora.state_dict(),
                    "head": head.state_dict() if head is not None else None,
                    "opt": opt.state_dict(),
                    "step": step,
                },
                resume,
            )
    io.shutdown()
    meta = {
        **meta_base,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "steps": o["steps"],
        "batch": o["batch"],
        "accum": accum,
        "lr": o["lr"],
        "replay_ratio": ratio,
        "seed": seed,
        "seconds_this_run": round(time.time() - t_start),
        "data": d["train"],
        "exclude": d.get("exclude", []),
    }
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    resume.unlink(missing_ok=True)
    print(f"saved {out / name}.pt", flush=True)
    print("done", flush=True)


def lr_at(step: int, o: dict) -> float:
    """Linear warmup, then cosine to ``min_lr_ratio`` of the peak."""
    import math

    if step < o["warmup"]:
        return o["lr"] * (step + 1) / o["warmup"]
    p = (step - o["warmup"]) / max(1, o["steps"] - o["warmup"])
    floor = o.get("min_lr_ratio", 0.1)
    return o["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def main() -> None:
    # shapes vary per batch; without this the cache fragments up to the gpu_mem_gb cap
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    train(sys.argv[1])


if __name__ == "__main__":
    main()
