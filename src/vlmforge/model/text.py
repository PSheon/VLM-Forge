"""Moondream 3.1 text model, region model and LoRA in plain PyTorch.

Vendored from VisionTraining's parity-verified ``src/syncvision/teacher/text.py`` (PRO 6000
engine fixtures: 99% greedy top-1 agreement). Compared with a plain transformer port:

- every forward is batched ``[B, T, D]`` with per-row positions;
- ``prefill`` / ``decode`` write and read a static KV cache, so greedy generation is
  one token per step and the decode step can be captured in a CUDA graph;
- inference never syncs with the host: prefill runs the MoE as grouped GEMMs over
  expert-sorted token-expert pairs, decode runs all 64 experts densely (cheaper than
  grouped GEMM at decode batch sizes, where reading the weights dominates); the autograd
  ``forward`` uses the per-expert loop by default, or the grouped GEMMs (with LoRA, the
  ``lora_train`` path) with ``moe="grouped"``.

Architecture: 24 parallel-residual blocks, tau-scaled attention, partial NeoX RoPE
(first 32 dims), prefix-LM mask (BOS + 729 image tokens bidirectional), dense
GELU-tanh MLP in layers 0-3, top-8-of-64 GeGLU MoE in layers 4-23 (FP8 experts
dequantised to BF16 at load).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention.bias import causal_lower_right

D, H, HD, L = 2048, 32, 64, 24
MOE_START, E, TOPK, I = 4, 64, 8, 1024  # noqa: E741  (expert inner dim)
PREFIX = 730  # BOS + 729 image tokens attend bidirectionally
ROPE_DIM, THETA = 32, 1.5e6
VOCAB = 51200


class KVCache:
    """Static per-layer K/V buffers ``[L, B, H, T, HD]``; row b position p lives at index p."""

    def __init__(self, batch: int, length: int, device, dtype=torch.bfloat16):
        # zeros, not empty: masked keys still enter P @ V, and 0 * NaN garbage = NaN.
        self.k = torch.zeros(L, batch, H, length, HD, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.batch, self.length = batch, length


def fp8_roundtrip(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantise to FP8 E4M3 with a per-tensor scale and back (the engine's KV-cache storage)."""
    q = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * scale).to(x.dtype)


class TextModel(nn.Module):
    """``kv_fp8``: round K/V through FP8 E4M3 with the checkpoint's per-layer scales, as the
    engine's KV cache does on SM 8.x-12.x (both its prefill and decode attention read the
    FP8 cache). Off by default here (training path); ``Moondream`` may turn it on."""

    def __init__(self, path: str, device: torch.device, dtype=torch.bfloat16, kv_fp8: bool = False):
        super().__init__()
        self.device, self.dtype, self.kv_fp8 = device, dtype, kv_fp8
        w = {}
        with safe_open(path, "pt", device=str(device)) as f:

            def get(k):
                return f.get_tensor(k)

            def bf(k):
                return get(k).to(dtype)

            w["wte"] = bf("text_model.transformer.embd.wte.weight")
            w["post_ln"] = (bf("text_model.lm_head.ln.weight"), bf("text_model.lm_head.ln.bias"))
            w["lm"] = (bf("text_model.lm_head.linear.weight"), bf("text_model.lm_head.linear.bias"))
            self.layers = []
            for i in range(L):
                p = f"text_model.transformer.h.{i}."
                lw = {
                    "ln": (bf(p + "ln.weight"), bf(p + "ln.bias")),
                    "qkv": (bf(p + "mixer.Wqkv.weight"), bf(p + "mixer.Wqkv.bias")),
                    "out": (bf(p + "mixer.out_proj.weight"), bf(p + "mixer.out_proj.bias")),
                    "tau_w": torch.cat([bf(p + "tau_wq"), bf(p + "tau_wv")], 0),  # [64, 6144]
                    "tau_alpha": get(p + "tau_alpha").float(),
                    "kv_scale": (
                        get(p + "kv_quantizer.k_scale").item(),
                        get(p + "kv_quantizer.v_scale").item(),
                    ),
                }
                if i < MOE_START:
                    lw["fc1"] = (bf(p + "mlp.fc1.weight"), bf(p + "mlp.fc1.bias"))
                    lw["fc2"] = (bf(p + "mlp.fc2.weight"), bf(p + "mlp.fc2.bias"))
                else:
                    lw["gate"] = (bf(p + "gate.weight"), bf(p + "gate.bias"))
                    up = (
                        get(p + "mlp.experts.weight").float()
                        * get(p + "moe_quant.up_scale")[:, :, None]
                    )
                    lw["w_up"] = up.to(dtype)  # [E, 2I, D], rows = [gate | up]
                    del up
                    down = (
                        get(p + "mlp.output_experts.weight").float()
                        * get(p + "moe_quant.down_scale")[:, :, None]
                    )
                    lw["w_down"] = down.to(dtype)  # [E, D, I]
                    del down
                self.layers.append(lw)
        self.w = w
        self.inv_freq = 1.0 / (
            THETA ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32) / ROPE_DIM)
        )
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.w["wte"])

    def logits(self, hidden):
        h = F.layer_norm(hidden, (D,), *self.w["post_ln"], eps=1e-5)
        return F.linear(h, *self.w["lm"])

    def _rope_tables(self, pos: torch.Tensor):
        """pos [B, T] -> cos, sin [B, T, 1, ROPE_DIM/2] (fp32) and log(pos+1) [B, T]."""
        freqs = pos[..., None].float() * self.inv_freq
        return freqs.cos()[..., None, :], freqs.sin()[..., None, :], torch.log(pos.float() + 1)

    @staticmethod
    def _rope(x, cos, sin):
        half = ROPE_DIM // 2
        x0, x1 = x[..., :half].float(), x[..., half:ROPE_DIM].float()
        o0 = x0 * cos - x1 * sin
        o1 = x1 * cos + x0 * sin
        return torch.cat([o0.to(x.dtype), o1.to(x.dtype), x[..., ROPE_DIM:]], -1)

    def _qkv(self, h, lw, cos, sin, logpos):
        """h [B, T, D] (normed) -> q, k, v [B, H, T, HD] with tau scaling and RoPE."""
        B, T, _ = h.shape
        qkv = F.linear(h, *lw["qkv"])
        tok = torch.tanh(F.linear(F.gelu(qkv, approximate="tanh"), lw["tau_w"]))  # [B, T, 64]
        tau_pos = (torch.sigmoid(logpos[..., None] * lw["tau_alpha"]) + 0.5).to(
            h.dtype
        )  # [B, T, H]
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, HD) * (tok[..., :H] + tau_pos)[..., None]
        v = v.view(B, T, H, HD) * (tok[..., H:] + tau_pos)[..., None]
        k = k.view(B, T, H, HD)
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        if self.kv_fp8:
            k, v = fp8_roundtrip(k, lw["kv_scale"][0]), fp8_roundtrip(v, lw["kv_scale"][1])
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _moe(self, h, lw, lora_moe):
        """Sparse per-expert loop (autograd-friendly, efficient for many tokens). h [N, D]."""
        T = h.shape[0]
        logits = F.linear(h, *lw["gate"])
        topv, topi = logits.topk(TOPK, dim=-1)
        topw = F.softmax(topv.float(), dim=-1).to(h.dtype)
        out = torch.zeros(T, D, dtype=h.dtype, device=h.device)
        for e in topi.unique().tolist():
            tok, slot = (topi == e).nonzero(as_tuple=True)
            xe = h[tok]
            u = xe @ lw["w_up"][e].t()
            if lora_moe is not None:
                u = u + (xe @ lora_moe.up_a[e].t()) @ lora_moe.up_b[e].t()
            g, up = u.split(I, dim=-1)
            a = F.gelu(g) * (up + 1)
            y = a @ lw["w_down"][e].t()
            if lora_moe is not None:
                y = y + (a @ lora_moe.down_a[e].t()) @ lora_moe.down_b[e].t()
            out = out.index_add(0, tok, y * topw[tok, slot, None])
        return out

    def _moe_dense(self, h, lw, lora_moe):
        """All 64 experts on every token, weighted by the (sparse) router weights. h [N, D].

        Same result as ``_moe`` up to summation order, with no host sync (CUDA-graph safe);
        used for decode with a LoRA (at decode batch sizes reading the expert weights
        dominates, so the 8x extra FLOPs are nearly free).
        """
        logits = F.linear(h, *lw["gate"])
        topv, topi = logits.topk(TOPK, dim=-1)
        topw = F.softmax(topv.float(), dim=-1).to(h.dtype)
        wfull = torch.zeros(h.shape[0], E, dtype=h.dtype, device=h.device).scatter_(1, topi, topw)
        hx = h[None].expand(E, -1, -1)
        u = torch.bmm(hx, lw["w_up"].transpose(1, 2))  # [E, N, 2I]
        if lora_moe is not None:
            u = u + torch.bmm(
                torch.bmm(hx, lora_moe.up_a.transpose(1, 2)), lora_moe.up_b.transpose(1, 2)
            )
        g, up = u.split(I, dim=-1)
        a = F.gelu(g) * (up + 1)
        y = torch.bmm(a, lw["w_down"].transpose(1, 2))  # [E, N, D]
        if lora_moe is not None:
            y = y + torch.bmm(
                torch.bmm(a, lora_moe.down_a.transpose(1, 2)), lora_moe.down_b.transpose(1, 2)
            )
        return (y * wfull.t()[:, :, None]).sum(0)

    def _moe_grouped(self, h, lw, lora_moe=None):
        """Token-expert pairs sorted by expert through ``torch._grouped_mm``. h [N, D].

        No host sync (CUDA-graph safe), deterministic, and per-expert products identical to
        ``_moe``; only the final 8-way weighted sum is accumulated in FP32 instead of BF16.
        Differentiable (grouped GEMM has a backward; in-place ops only without grad), so it is
        also the fast training path. LoRA expert ranks (r/8) are zero-padded to a multiple of 8
        to meet grouped GEMM alignment.
        """
        N = h.shape[0]
        logits = F.linear(h, *lw["gate"])
        topv, topi = logits.topk(TOPK, dim=-1)
        topw = F.softmax(topv.float(), dim=-1).to(h.dtype)
        flat = topi.reshape(-1)
        order = flat.argsort(stable=True)
        offs = torch.searchsorted(flat[order], torch.arange(E, device=h.device), right=True).to(
            torch.int32
        )
        xs = h.index_select(0, order // TOPK)
        u = torch._grouped_mm(xs, lw["w_up"].transpose(1, 2), offs=offs)  # [N*8, 2I]
        if lora_moe is not None:
            u = u + _grouped_lora(xs, lora_moe.up_a, lora_moe.up_b, offs)
        g, up = u.split(I, dim=-1)
        if torch.is_grad_enabled():
            a = F.gelu(g) * (up + 1)
        else:
            a = F.gelu(g).mul_(up.add_(1))
        y = torch._grouped_mm(a, lw["w_down"].transpose(1, 2), offs=offs)  # [N*8, D]
        if lora_moe is not None:
            y = y + _grouped_lora(a, lora_moe.down_a, lora_moe.down_b, offs)
        w = topw.reshape(-1)[order, None]
        y = y * w if torch.is_grad_enabled() else y.mul_(w)
        inv = torch.empty_like(order).scatter_(
            0, order, torch.arange(order.numel(), device=h.device)
        )
        return y.index_select(0, inv).view(N, TOPK, D).sum(1)  # back to token-major order

    def _dense(self, h, lw, lora_dense):
        a = F.linear(h, *lw["fc1"])
        if lora_dense is not None:
            a = a + (h @ lora_dense.up_a.t()) @ lora_dense.up_b.t()
        a = F.gelu(a, approximate="tanh")
        y = F.linear(a, *lw["fc2"])
        if lora_dense is not None:
            y = y + (a @ lora_dense.down_a.t()) @ lora_dense.down_b.t()
        return y

    def _mlp(self, h, i, lw, lora, moe: str):
        """h [B, T, D] -> [B, T, D]. moe: "loop" (per-expert, autograd), "grouped" (grouped
        GEMM, autograd; the LoRA training path) or "fast" (inference: dense experts for
        decode, grouped GEMM for prefill; no host sync)."""
        B, T, _ = h.shape
        if i < MOE_START:
            return self._dense(h, lw, lora.dense[i] if lora is not None else None)
        x = h.reshape(B * T, D)
        lm = lora.moe[i - MOE_START] if lora is not None else None
        if moe == "fast" and T == 1:  # decode: dense beats grouped GEMM at small token counts
            return self._moe_dense(x, lw, lm).view(B, T, D)
        if moe in ("fast", "grouped"):
            return self._moe_grouped(x, lw, lm).view(B, T, D)
        return self._moe(x, lw, lm).view(B, T, D)

    # ------------------------------------------------------------------ forward
    def _run(self, x, pos, attend, lora=None, moe="loop", checkpoint=False):
        """Shared layer loop. ``attend(i, q, k, v) -> o [B, T, D]`` does cache I/O + attention."""
        cos, sin, logpos = self._rope_tables(pos)
        for i, lw in enumerate(self.layers):

            def block(x, i=i, lw=lw):
                h = F.layer_norm(x, (D,), *lw["ln"], eps=1e-5)
                q, k, v = self._qkv(h, lw, cos, sin, logpos)
                o = F.linear(attend(i, q, k, v), *lw["out"])
                return x + o + self._mlp(h, i, lw, lora, moe)

            if checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    @staticmethod
    def prefix_mask(T: int, device) -> torch.Tensor:
        q_i = torch.arange(T, device=device)[:, None]
        k_i = torch.arange(T, device=device)[None, :]
        return (k_i <= q_i) | ((q_i < PREFIX) & (k_i < PREFIX))

    @staticmethod
    def prefix_attention(q, k, v):
        """Prefix-LM attention for positions 0..T-1 (T > PREFIX), q/k/v [B, H, T, HD]: the
        first PREFIX positions attend to each other bidirectionally, the rest causally. Two
        unmasked / lower-right-causal SDPA calls instead of one with an explicit mask
        (same result, ~7x faster)."""
        T = q.shape[2]
        o1 = F.scaled_dot_product_attention(q[:, :, :PREFIX], k[:, :, :PREFIX], v[:, :, :PREFIX])
        o2 = F.scaled_dot_product_attention(
            q[:, :, PREFIX:], k, v, attn_mask=causal_lower_right(T - PREFIX, T)
        )
        return torch.cat([o1, o2], 2)

    def forward(self, x: torch.Tensor, lora=None, checkpoint: bool = False, moe: str = "loop"):
        """Autograd path. x: [T, D] or [B, T, D] input embeddings (BOS, 729 image tokens, prompt,
        targets). Returns hidden states (pre final norm) of the same shape. ``moe="grouped"``
        runs the experts as grouped GEMMs (same products, much faster for training)."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x[None]
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).expand(B, T)
        mask = self.prefix_mask(T, x.device) if T <= PREFIX else None

        def attend(i, q, k, v):
            if mask is None:
                o = self.prefix_attention(q, k, v)
            else:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return o.transpose(1, 2).reshape(B, T, D)

        out = self._run(x, pos, attend, lora, moe=moe, checkpoint=checkpoint)
        return out[0] if squeeze else out

    @torch.no_grad()
    def prefill(self, x: torch.Tensor, cache: KVCache, rows: list[int], lora=None):
        """x [b, T, D] (right-padded rows, positions 0..T-1) -> hidden [b, T, D].

        Writes K/V for positions 0..T-1 into cache rows ``rows``. Padding positions get
        junk K/V; decode overwrites them before any real token can see them.
        """
        b, T, _ = x.shape
        assert T > PREFIX and len(rows) == b
        pos = torch.arange(T, device=x.device).expand(b, T)
        span = rows == list(range(rows[0], rows[0] + b))  # slice copy; per-row copies otherwise

        def attend(i, q, k, v):
            for cache_t, new in ((cache.k[i], k), (cache.v[i], v)):
                if span:
                    cache_t[rows[0] : rows[0] + b, :, :T].copy_(new)
                else:
                    for j, r in enumerate(rows):
                        cache_t[r, :, :T].copy_(new[j])
            return self.prefix_attention(q, k, v).transpose(1, 2).reshape(b, T, D)

        return self._run(x, pos, attend, lora, moe="fast")

    @torch.no_grad()
    def decode(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        cache: KVCache,
        lora=None,
        kv_len: int | None = None,
    ):
        """One token per row: x [B, D], pos [B] (long) -> hidden [B, D]. Writes K/V at ``pos``.

        Attention reads only the first ``kv_len`` cache positions (must exceed every
        active row's ``pos``); rows past it are idle and their output is junk. It is two
        batched GEMVs with flash-style numerics (FP32 scores and normaliser, unnormalised
        BF16 probabilities): same error as SDPA, ~2x faster than its masked decode kernel.
        """
        B = x.shape[0]
        kv_len = cache.length if kv_len is None else kv_len
        rows = torch.arange(B, device=x.device)
        valid = (torch.arange(kv_len, device=x.device)[None, :] <= pos[:, None])[
            :, None, :
        ]  # [B, 1, kv]
        scale = HD**-0.5

        def attend(i, q, k, v):
            cache.k[i, rows, :, pos] = k[:, :, 0]
            cache.v[i, rows, :, pos] = v[:, :, 0]
            kc = cache.k[i, :, :, :kv_len].reshape(B * H, kv_len, HD)
            vc = cache.v[i, :, :, :kv_len].reshape(B * H, kv_len, HD)
            s = torch.bmm(
                q.reshape(B * H, 1, HD), kc.transpose(1, 2), out_dtype=torch.float32
            ).view(B, H, kv_len)
            s = s.mul_(scale).masked_fill_(~valid, float("-inf"))
            p = (s - s.amax(-1, keepdim=True)).exp_()
            o = torch.bmm(p.to(x.dtype).view(B * H, 1, kv_len), vc, out_dtype=torch.float32).view(
                B, H, HD
            )
            return (o / p.sum(-1, keepdim=True)).to(x.dtype).view(B, 1, D)

        return self._run(x[:, None], pos[:, None], attend, lora, moe="fast")[:, 0]


def _grouped_lora(x, a, b, offs):
    """Per-expert LoRA ``(x @ a[e].T) @ b[e].T`` over expert-sorted rows; a [E, r, Din],
    b [E, Dout, r]. r is zero-padded to a multiple of 8 (16-byte grouped GEMM alignment)."""
    pad = -a.shape[1] % 8
    if pad:
        a, b = F.pad(a, (0, 0, 0, pad)), F.pad(b, (0, pad))
    t = torch._grouped_mm(x, a.transpose(1, 2), offs=offs)
    return torch._grouped_mm(t, b.transpose(1, 2), offs=offs)


COORD_BINS = 1024
SIZE_BINS = 1024


class RegionModel:
    """Coordinate / size encoders (input side) and decoders (output side) for detect and point.

    A coordinate x enters as coord_encoder(fourier(x)) in place of token 5; a box size
    (w, h) as size_encoder(fourier(w, h)) in place of token 6. When the LM predicts
    token 5/6, the value is decoded from the same (pre-final-norm) hidden state. Bin
    values come from BF16 lookup tables, exactly like the engine.
    """

    def __init__(self, path: str, device, dtype=torch.bfloat16):
        with safe_open(path, "pt", device=str(device)) as f:
            g = lambda k: f.get_tensor("region_model." + k).to(dtype)  # noqa: E731
            self.coord_features = g("coordinate_features.weight").t()  # [1, 128]
            self.size_features = g("size_features.weight").t()  # [2, 256]
            self.coord_enc = (g("coordinate_encoder.weight"), g("coordinate_encoder.bias"))
            self.size_enc = (g("size_encoder.weight"), g("size_encoder.bias"))
            self.ln = (g("ln.weight"), g("ln.bias"))
            self.dec_w = torch.cat(
                [g("coordinate_head.weight"), g("size_head.weight")], 0
            )  # [3072, D]
            self.dec_b = torch.cat([g("coordinate_head.bias"), g("size_head.bias")], 0)
        self.coord_lut = torch.linspace(
            0.0, 1.0, COORD_BINS, device=device, dtype=torch.float32
        ).to(dtype)
        self.size_lut = torch.exp2(
            torch.linspace(-10.0, 0.0, SIZE_BINS, device=device, dtype=torch.float32)
        ).to(dtype)

    @staticmethod
    def _fourier(x, w):
        f = 2 * math.pi * x @ w
        return torch.cat([f.cos(), f.sin()], -1)

    def encode_coord(self, x: torch.Tensor) -> torch.Tensor:  # x: [n] -> [n, D]
        return F.linear(
            self._fourier(x[:, None].to(self.coord_features.dtype), self.coord_features),
            *self.coord_enc,
        )

    def encode_size(self, wh: torch.Tensor) -> torch.Tensor:  # wh: [n, 2] -> [n, D]
        return F.linear(
            self._fourier(wh.to(self.size_features.dtype), self.size_features), *self.size_enc
        )

    def decode(self, hidden: torch.Tensor, head: "RegionHead | None" = None):
        """hidden [n, D] -> logits [n, 1024] for coord, width and height.

        ``head``: a trainable :class:`RegionHead` (GOALS §5) whose cast replaces the frozen
        decoder tensors for this call."""
        ln, w, b = (self.ln, self.dec_w, self.dec_b) if head is None else head.view(hidden.dtype)
        logits = F.linear(F.layer_norm(hidden, (D,), *ln, eps=1e-5), w, b)
        return (
            logits[:, :COORD_BINS],
            logits[:, COORD_BINS : COORD_BINS + SIZE_BINS],
            logits[:, COORD_BINS + SIZE_BINS :],
        )

    def decode_values(self, hidden: torch.Tensor):
        """Greedy values: hidden [n, D] -> coord [n], size [n, 2] (BF16 LUT values)."""
        c, w, h = self.decode(hidden)
        size = torch.stack([self.size_lut[w.argmax(-1)], self.size_lut[h.argmax(-1)]], -1)
        return self.coord_lut[c.argmax(-1)], size

    @staticmethod
    def coord_bin(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(0, 1) * (COORD_BINS - 1)).round().long()

    @staticmethod
    def size_bin(s: torch.Tensor) -> torch.Tensor:
        return ((torch.log2(s.clamp(2**-10, 1.0)) + 10) / 10 * (SIZE_BINS - 1)).round().long()

    # upstream tensor names of the decoder side, as stored in adapters and model files
    HEAD_KEYS = (
        "region_model.ln.weight",
        "region_model.ln.bias",
        "region_model.coordinate_head.weight",
        "region_model.coordinate_head.bias",
        "region_model.size_head.weight",
        "region_model.size_head.bias",
    )

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """The decoder side (layer norm + coordinate / size heads) under upstream names: what
        the UI engine keeps to restore the base after an adapter's ``load_head``."""
        parts = (
            self.ln[0],
            self.ln[1],
            self.dec_w[:COORD_BINS],
            self.dec_b[:COORD_BINS],
            self.dec_w[COORD_BINS:],
            self.dec_b[COORD_BINS:],
        )
        return dict(zip(self.HEAD_KEYS, parts, strict=True))

    def load_head(self, sd: dict[str, torch.Tensor]) -> None:
        """Replace the decoder side with a fine-tuned one (an adapter's ``region_head``)."""
        dt, dev = self.dec_w.dtype, self.dec_w.device
        g = lambda k: sd[k].to(dev, dt)  # noqa: E731
        self.ln = (g(self.HEAD_KEYS[0]), g(self.HEAD_KEYS[1]))
        self.dec_w = torch.cat([g(self.HEAD_KEYS[2]), g(self.HEAD_KEYS[4])], 0)
        self.dec_b = torch.cat([g(self.HEAD_KEYS[3]), g(self.HEAD_KEYS[5])], 0)


class RegionHead(nn.Module):
    """Trainable copy of the region decoder (layer norm + coordinate / size heads): FP32
    masters initialised from the frozen :class:`RegionModel`. GOALS §5: the region head is
    trainable, the input-side coordinate / size encoders stay frozen."""

    def __init__(self, region: RegionModel):
        super().__init__()
        p = lambda t: nn.Parameter(t.detach().float().clone())  # noqa: E731
        self.ln_w, self.ln_b = p(region.ln[0]), p(region.ln[1])
        self.dec_w, self.dec_b = p(region.dec_w), p(region.dec_b)

    def view(self, dtype):
        return (
            (self.ln_w.to(dtype), self.ln_b.to(dtype)),
            self.dec_w.to(dtype),
            self.dec_b.to(dtype),
        )

    def state_dict_upstream(self, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
        """The fine-tuned decoder under upstream tensor names (what an adapter stores)."""
        w, b = self.dec_w.detach().to(dtype), self.dec_b.detach().to(dtype)
        parts = (
            self.ln_w.detach().to(dtype),
            self.ln_b.detach().to(dtype),
            w[:COORD_BINS],
            b[:COORD_BINS],
            w[COORD_BINS:],
            b[COORD_BINS:],
        )
        return dict(zip(RegionModel.HEAD_KEYS, parts, strict=True))


class TextLoRA(nn.Module):
    """LoRA in the inference engine's TextLoRA layout.

    dense.{0..3}: up_a [r, D], up_b [4D, r], down_a [r, 4D], down_b [D, r]
    moe.{0..19}:  up_a [E, r/8, D], up_b [E, 2I, r/8], down_a [E, r/8, I], down_b [E, D, r/8]
    B matrices start at zero so the adapter begins as a no-op.

    The engine has no alpha: ``scale = alpha / rank`` (GOALS §5: alpha 32 at rank 16) is
    applied by the trainer's forward view and folded into the B matrices by
    :meth:`engine_state_dict`, so a saved adapter loads into the engine unchanged.
    """

    def __init__(self, rank: int, device, dtype=torch.bfloat16, alpha: float | None = None):
        super().__init__()
        assert rank % 8 == 0
        re = rank // 8
        self.alpha = float(rank if alpha is None else alpha)
        self.scale = self.alpha / rank

        def a(*shape):
            t = torch.empty(*shape, device=device, dtype=torch.float32)
            nn.init.kaiming_uniform_(t, a=math.sqrt(5))
            return nn.Parameter(t.to(dtype))

        def z(*shape):
            return nn.Parameter(torch.zeros(*shape, device=device, dtype=dtype))

        self.rank = rank
        self.dense = nn.ModuleList()
        for _ in range(MOE_START):
            m = nn.Module()
            m.up_a, m.up_b, m.down_a, m.down_b = (
                a(rank, D),
                z(4 * D, rank),
                a(rank, 4 * D),
                z(D, rank),
            )
            self.dense.append(m)
        self.moe = nn.ModuleList()
        for _ in range(L - MOE_START):
            m = nn.Module()
            m.up_a, m.up_b, m.down_a, m.down_b = (
                a(E, re, D),
                z(E, 2 * I, re),
                a(E, re, I),
                z(E, D, re),
            )
            self.moe.append(m)

    def engine_state_dict(self):
        """Engine layout with the alpha scale folded into the B matrices."""
        sd = {}
        for i, m in enumerate(self.dense):
            for n in ("up_a", "up_b", "down_a", "down_b"):
                t = getattr(m, n).detach().clone()
                sd[f"text_lora.dense.{i}.{n}"] = t * self.scale if n.endswith("_b") else t
        for j, m in enumerate(self.moe):
            for n in ("up_a", "up_b", "down_a", "down_b"):
                t = getattr(m, n).detach().clone()
                sd[f"text_lora.moe.{j}.{n}"] = t * self.scale if n.endswith("_b") else t
        return sd

    @classmethod
    def from_checkpoint(cls, path: str, device, dtype=torch.bfloat16) -> "TextLoRA":
        """Load a ``{"lora_state_dict", "rank"}`` adapter checkpoint (engine layout)."""
        ckpt = torch.load(path, map_location="cpu")
        lora = cls(int(ckpt["rank"]), device, dtype)
        sd = ckpt["lora_state_dict"]
        with torch.no_grad():
            for i, m in enumerate(lora.dense):
                for n in ("up_a", "up_b", "down_a", "down_b"):
                    getattr(m, n).copy_(sd[f"text_lora.dense.{i}.{n}"])
            for j, m in enumerate(lora.moe):
                for n in ("up_a", "up_b", "down_a", "down_b"):
                    getattr(m, n).copy_(sd[f"text_lora.moe.{j}.{n}"])
        return lora.requires_grad_(False)
