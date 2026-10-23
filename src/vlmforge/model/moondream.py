"""Batched greedy Moondream 3.1 inference: caption / query / detect / point.

    t = Moondream(weights, tokenizer_dir)
    t.caption(images, "long")            -> [str]
    t.query(images, ["How many ...?"])   -> [str]
    t.detect(images, "person")           -> [[{"x_min", "y_min", "x_max", "y_max"}]]   (0-1)
    t.point(images, "person")            -> [[{"x", "y"}]]                             (0-1)

``images`` is a list of PIL images / paths / bytes / HWC uint8 arrays, or precomputed
image tokens (a [N, 729, 2048] tensor or a list of [729, 2048] tensors, see ``encode``).

Generation runs on a fixed number of decode slots sharing one static KV cache.
Finished rows are refilled from the queue (continuous batching); the per-token
step (LM head, skill token mask, region decode, embedding of the next token, one
decoder forward) has no host sync and is replayed as a CUDA graph.

Skill semantics follow the engine: greedy; detect allows {coord, eos} -> {coord} ->
{size} per object, max 50 objects (3 * 50 + 1 tokens) by default; point allows
{coord, eos} -> {coord}; caption / query stop at EOS or ``max_tokens`` (768).
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .text import D, KVCache, RegionModel, TextModel
from .vision import VisionEncoder

BOS, EOS, ANSWER, COORD_ID, SIZE_ID = 0, 0, 3, 5, 6
CAPTION_LENGTH = {"short": 12492, "normal": 6382, "long": 4059}
DEFAULT_MAX_TOKENS = 768
DEFAULT_MAX_OBJECTS = 50
IMAGE_TOKENS = 729

# Decode states (row of the token mask table) and their transitions.
FREE, DET_X, DET_Y, DET_SIZE, PT_X, PT_Y = range(6)
SKILL_STATE = {"caption": FREE, "query": FREE, "detect": DET_X, "point": PT_X}


def prompt_ids(tokenizer: Tokenizer, skill: str, item: dict) -> list[int]:
    """Full prompt including BOS; the image goes right after BOS (engine templates)."""
    if skill == "query":
        return [BOS, 1, 15381, 2] + tokenizer.encode(item["question"]).ids + [ANSWER]
    if skill == "caption":
        return [BOS, 1, 32708, 2, CAPTION_LENGTH[item.get("length", "normal")], ANSWER]
    if skill == "detect":
        return (
            [BOS, 1, 7235, 476, 2] + tokenizer.encode(" " + item["object"].strip()).ids + [ANSWER]
        )
    if skill == "point":
        return [BOS, 1, 2581, 2] + tokenizer.encode(" " + item["object"].strip()).ids + [ANSWER]
    raise ValueError(f"unknown skill {skill!r}")


@dataclass
class Request:
    feat: torch.Tensor  # [729, D] image tokens
    prompt: list[int]  # including BOS
    state: int  # initial decode state (SKILL_STATE)
    max_new: int


@dataclass
class Generation:
    tokens: list[int]  # generated ids, including the final EOS when the row stopped on it
    values: list[tuple[float, float]] = field(
        default_factory=list
    )  # per token: (coord, 0) / (w, h)


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


class _Decoder:
    """Fixed slots + static KV cache; ``step`` is graph-captured per attention-window bucket."""

    BUCKET = 128

    def __init__(
        self,
        text: TextModel,
        region: RegionModel,
        slots: int,
        length: int,
        max_new: int,
        lora=None,
        cuda_graph: bool = True,
    ):
        dev = text.device
        self.text, self.region, self.lora = text, region, lora
        self.B, self.length, self.M = slots, length, max_new
        self.cache = KVCache(slots, length, dev)
        z = lambda *s, dt=torch.long: torch.zeros(*s, device=dev, dtype=dt)  # noqa: E731
        self.pos, self.state, self.count, self.maxnew = z(slots), z(slots), z(slots), z(slots)
        self.done = torch.ones(slots, device=dev, dtype=torch.bool)
        self.h = z(slots, D, dt=text.dtype)
        self.toks = z(slots, max_new)
        self.vals = z(slots, max_new, 2, dt=torch.float32)
        self.rows = torch.arange(slots, device=dev)
        V = text.w["lm"][0].shape[0]
        allowed = {
            DET_X: [COORD_ID, EOS],
            DET_Y: [COORD_ID],
            DET_SIZE: [SIZE_ID],
            PT_X: [COORD_ID, EOS],
            PT_Y: [COORD_ID],
        }
        self.mask_table = torch.zeros(6, V, device=dev)
        for s, ids in allowed.items():
            self.mask_table[s] = float("-inf")
            self.mask_table[s, ids] = 0.0
        # next_state[state, cls]; cls 0 = other token, 1 = coord, 2 = size
        nxt = [[s, s, s] for s in range(6)]
        nxt[DET_X][1], nxt[DET_Y][1], nxt[DET_SIZE][2] = DET_Y, DET_SIZE, DET_X
        nxt[PT_X][1], nxt[PT_Y][1] = PT_Y, PT_X
        self.next_state = torch.tensor(nxt, device=dev)
        self.cuda_graph = cuda_graph
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle() if cuda_graph else None

    # ---------------------------------------------------------------- device side
    def _step(self, kv_len: int):
        h, active = self.h, ~self.done
        logits = self.text.logits(h).float() + self.mask_table[self.state]
        tok = logits.argmax(-1)
        coord, size = self.region.decode_values(h)
        is_coord, is_size = tok == COORD_ID, tok == SIZE_ID
        idx = self.count.clamp(max=self.M - 1)
        val = torch.where(
            is_size[:, None],
            size.float(),
            torch.stack([coord.float(), torch.zeros_like(coord, dtype=torch.float32)], -1),
        )
        self.toks[self.rows, idx] = torch.where(active, tok, self.toks[self.rows, idx])
        self.vals[self.rows, idx] = torch.where(active[:, None], val, self.vals[self.rows, idx])
        self.count.add_(active.long())
        cls = is_coord.long() + 2 * is_size.long()
        self.state.copy_(torch.where(active, self.next_state[self.state, cls], self.state))
        self.done.logical_or_(active & ((tok == EOS) | (self.count >= self.maxnew)))
        emb = self.text.embed(tok)
        emb = torch.where(is_coord[:, None], self.region.encode_coord(coord), emb)
        emb = torch.where(is_size[:, None], self.region.encode_size(size), emb)
        self.h.copy_(self.text.decode(emb, self.pos, self.cache, self.lora, kv_len=kv_len))
        self.pos.add_((~self.done).long())

    def step(self, kv_len: int):
        if not self.cuda_graph:
            return self._step(kv_len)
        g = self.graphs.get(kv_len)
        if g is None:
            # Warm up eagerly on a side stream, then restore the (small) state: the cache
            # writes the warm-up made are rewritten identically by the real step.
            names = ("pos", "state", "count", "done", "h", "toks", "vals")
            saved = [getattr(self, n).clone() for n in names]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._step(kv_len)
            torch.cuda.current_stream().wait_stream(s)
            for n, t in zip(names, saved, strict=True):
                getattr(self, n).copy_(t)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._step(kv_len)
            self.graphs[kv_len] = g
        g.replay()

    @torch.no_grad()
    def admit(self, slots: list[int], reqs: list[Request]):
        """Prefill ``reqs`` into ``slots`` (one batched prefill, rows right-padded)."""
        dev = self.text.device
        lens = [IMAGE_TOKENS + len(r.prompt) for r in reqs]
        T = max(lens)
        x = torch.zeros(len(reqs), T, D, device=dev, dtype=self.text.dtype)
        for j, r in enumerate(reqs):
            ids = torch.tensor(r.prompt, device=dev)
            emb = self.text.embed(ids)
            x[j, 0] = emb[0]
            x[j, 1 : 1 + IMAGE_TOKENS] = r.feat
            x[j, 1 + IMAGE_TOKENS : lens[j]] = emb[1:]
        hidden = self.text.prefill(x, self.cache, slots, self.lora)
        rows = torch.tensor(slots, device=dev)
        lens_t = torch.tensor(lens, device=dev)
        self.h[rows] = hidden[torch.arange(len(reqs), device=dev), lens_t - 1]
        self.pos[rows] = lens_t
        self.state[rows] = torch.tensor([r.state for r in reqs], device=dev)
        self.maxnew[rows] = torch.tensor([r.max_new for r in reqs], device=dev)
        self.count[rows] = 0
        self.done[rows] = False

    @torch.no_grad()
    def run(
        self, reqs: list[Request], sync_every: int = 8, prefill_rows: int = 16
    ) -> list[Generation]:
        out: list[Generation | None] = [None] * len(reqs)
        queue = deque(range(len(reqs)))
        owner = [-1] * self.B  # request index per slot
        pos_host = [0] * self.B
        while True:
            free = [s for s in range(self.B) if owner[s] < 0]
            busy = self.B - len(free)
            if queue and (busy == 0 or len(free) >= max(1, self.B // 8)):
                take = free[: len(queue)]
                for c in range(0, len(take), prefill_rows):
                    chunk = take[c : c + prefill_rows]
                    idx = [queue.popleft() for _ in chunk]
                    self.admit(chunk, [reqs[i] for i in idx])
                    for s, i in zip(chunk, idx, strict=True):
                        owner[s] = i
                        pos_host[s] = IMAGE_TOKENS + len(reqs[i].prompt)
            if all(o < 0 for o in owner):
                break
            need = max(pos_host[s] for s in range(self.B) if owner[s] >= 0) + sync_every + 1
            kv_len = min(self.length, _round_up(need, self.BUCKET))
            for _ in range(sync_every):
                self.step(kv_len)
            done = self.done.tolist()
            pos_host = self.pos.tolist()
            fin = [s for s in range(self.B) if owner[s] >= 0 and done[s]]
            if fin:
                counts = self.count.tolist()
                toks, vals = self.toks.cpu(), self.vals.cpu()
                for s in fin:
                    n = counts[s]
                    out[owner[s]] = Generation(
                        toks[s, :n].tolist(), [tuple(v) for v in vals[s, :n].tolist()]
                    )
                    owner[s] = -1
        return out


class Moondream:
    def __init__(
        self,
        weights: str | Path,
        tokenizer: str | Path,
        device: str = "cuda",
        batch_size: int = 32,
        cuda_graph: bool = True,
        lora=None,
        region_head: dict | None = None,
        kv_fp8: bool = False,
    ):
        """``lora``: a :class:`TextLoRA`; ``region_head``: an adapter's fine-tuned region
        decoder (upstream tensor names, see ``RegionModel.load_head``)."""
        self.device = torch.device(device)
        weights = str(weights)
        tok_path = Path(tokenizer)
        self.tokenizer = Tokenizer.from_file(
            str(tok_path / "tokenizer.json" if tok_path.is_dir() else tok_path)
        )
        self.vision = VisionEncoder(weights, self.device)
        self.text = TextModel(weights, self.device, kv_fp8=kv_fp8)
        self.region = RegionModel(weights, self.device)
        if region_head:
            self.region.load_head(region_head)
        self.batch_size, self.cuda_graph, self.lora = batch_size, cuda_graph, lora
        self._decoder: _Decoder | None = None

    # ---------------------------------------------------------------- low level
    def encode(self, images) -> torch.Tensor:
        """Images -> image tokens [N, 729, 2048]; tensors are passed through."""
        if isinstance(images, torch.Tensor):
            return images.to(self.device, self.text.dtype)
        if images and all(isinstance(im, torch.Tensor) for im in images):
            return torch.stack([im.to(self.device, self.text.dtype) for im in images])
        return self.vision(images)

    def _decoder_for(self, n: int, length: int, max_new: int) -> _Decoder:
        slots = min(self.batch_size, n)
        length = _round_up(length, _Decoder.BUCKET)
        d = self._decoder
        if (
            d is None
            or d.B != slots
            or d.length < length
            or d.M < max_new
            or d.lora is not self.lora
        ):
            self._decoder = d = None
            torch.cuda.empty_cache()
            d = self._decoder = _Decoder(
                self.text, self.region, slots, length, max_new, self.lora, self.cuda_graph
            )
        return d

    @torch.no_grad()
    def generate(self, reqs: list[Request]) -> list[Generation]:
        if not reqs:
            return []
        max_new = max(r.max_new for r in reqs)
        length = max(IMAGE_TOKENS + len(r.prompt) for r in reqs) + max_new + 1
        return self._decoder_for(len(reqs), length, max_new).run(reqs)

    def _run(self, images, skill: str, items: list[dict], max_new: int) -> list[Generation]:
        if len(items) != len(images):
            raise ValueError(f"{len(images)} images but {len(items)} prompts")
        chunk = 8 * self.batch_size  # bounds the image tokens held on the GPU
        out = []
        for i in range(0, len(images), chunk):
            feats = self.encode(images[i : i + chunk])
            reqs = [
                Request(f, prompt_ids(self.tokenizer, skill, it), SKILL_STATE[skill], max_new)
                for f, it in zip(feats, items[i : i + chunk], strict=True)
            ]
            out += self.generate(reqs)
        return out

    @staticmethod
    def _per_image(value, n: int) -> list:
        return [value] * n if isinstance(value, str) else list(value)

    def _text(self, g: Generation) -> str:
        ids = g.tokens[:-1] if g.tokens and g.tokens[-1] == EOS else g.tokens
        return self.tokenizer.decode(ids)

    # ---------------------------------------------------------------- skills
    def caption(
        self, images, length: str = "normal", max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> list[str]:
        n = len(images)
        gens = self._run(images, "caption", [{"length": length}] * n, max_tokens)
        return [self._text(g) for g in gens]

    def query(self, images, questions, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[str]:
        qs = self._per_image(questions, len(images))
        gens = self._run(images, "query", [{"question": q} for q in qs], max_tokens)
        return [self._text(g) for g in gens]

    def detect(self, images, objects, max_objects: int = DEFAULT_MAX_OBJECTS) -> list[list[dict]]:
        objs = self._per_image(objects, len(images))
        gens = self._run(images, "detect", [{"object": o} for o in objs], 3 * max_objects + 1)
        return [boxes_from(g) for g in gens]

    def point(self, images, objects, max_objects: int | None = None) -> list[list[dict]]:
        objs = self._per_image(objects, len(images))
        max_new = DEFAULT_MAX_TOKENS if max_objects is None else max(2 * max_objects + 1, 2)
        gens = self._run(images, "point", [{"object": o} for o in objs], max_new)
        return [points_from(g) for g in gens]


def boxes_from(g: Generation) -> list[dict]:
    """Detect stream -> normalized boxes clamped to [0, 1] (as the engine does)."""
    out, px, py = [], None, None
    for t, (a, b) in zip(g.tokens, g.values, strict=True):
        if t == COORD_ID:
            if px is None:
                px = a
            elif py is None:
                py = a
            else:
                px, py = a, None
        elif t == SIZE_ID and px is not None and py is not None:
            out.append(
                {
                    "x_min": max(px - a / 2, 0.0),
                    "y_min": max(py - b / 2, 0.0),
                    "x_max": min(px + a / 2, 1.0),
                    "y_max": min(py + b / 2, 1.0),
                }
            )
            px = py = None
    return out


def points_from(g: Generation) -> list[dict]:
    out, px = [], None
    for t, (a, _) in zip(g.tokens, g.values, strict=True):
        if t == COORD_ID:
            if px is None:
                px = a
            else:
                out.append({"x": px, "y": a})
                px = None
    return out
