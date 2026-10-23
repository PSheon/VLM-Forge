"""Moondream 3.1 image encoder in plain PyTorch: tiling, SigLIP-style encoder, projection.

Follows the upstream reference (moondream 3-preview ``vision.py`` / ``image_crops.py``,
BSL 1.1, internal use) and the 3.1 engine's tiling:

- crop 0 is the whole image Lanczos-resized to 378 x 378; then up to 12 local crops
  of 378 px cut from the image Lanczos-resized to ``tiles * 266 + 112`` px, with a
  4-patch (56 px) overlap margin. A (1, 1) tiling still yields one local crop.
- pixels ``(x / 255 - 0.5) / 0.5`` in BF16; 14 px patches; 27 pre-LN blocks
  (dim 1152, 16 heads x 72, GELU-tanh MLP 4304), learned position embedding.
- local features are stitched (each tile keeps its interior plus the outer image
  border), adaptive-avg-pooled to 27 x 27, concatenated with the global features
  -> [729, 2304] -> proj_mlp (2304 -> 8192 -> 2048, GELU-tanh).

Resampling reproduces the engine's native Lanczos3 (Rust fast_image_resize) bit for
bit, in integer torch ops on the GPU; PIL's Lanczos differs by up to ~150 levels on
large downscales and the encoder is very sensitive to that. JPEG decoding uses PIL
(libjpeg-turbo) while the engine uses a Rust decoder: pixels differ by up to ~4
levels, which lowers the feature cosine slightly (see ``tools/teacher_parity.py``).
"""

import io
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

CROP, PATCH, MARGIN, MAX_CROPS = 378, 14, 4, 12
GRID = CROP // PATCH  # 27
ENC_DIM, ENC_HEADS, ENC_LAYERS = 1152, 16, 27
WINDOW = (GRID - 2 * MARGIN) * PATCH  # 266 px usable per local crop
MARGIN_PX = MARGIN * PATCH  # 56


def load_rgb(image) -> np.ndarray:
    """PIL image / path / bytes / HWC uint8 array -> HWC uint8 RGB.

    Alpha is dropped and EXIF orientation ignored, like the engine."""
    if isinstance(image, np.ndarray):
        a = image
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        return np.ascontiguousarray(np.clip(a, 0, 255).astype(np.uint8)[..., :3])
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        image = Image.open(image)
    return np.array(image.convert("RGB"))  # RGBA -> RGB drops alpha, like the engine


def select_tiling(height: int, width: int, crop_size: int, max_crops: int) -> tuple[int, int]:
    if height <= crop_size or width <= crop_size:
        return (1, 1)
    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)
    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))
    h_tiles = max(math.floor(math.sqrt(max_crops * height / width)), min_h)
    w_tiles = max(math.floor(math.sqrt(max_crops * width / height)), min_w)
    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)
    return (max(1, h_tiles), max(1, w_tiles))


# ---------------------------------------------------------------------- exact Lanczos3
_MAX_PREC = 22  # the i16 coefficient limit binds first
_COEFFS: dict = {}


def _axis_coeffs(n_in: int, n_out: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pillow-style Lanczos3 taps for one axis, quantised like fast_image_resize's U8 path.

    Returns tap indices [T, n_out], integer weights [T, n_out] and the fixed-point
    precision: the largest p (< 22) with round(max_weight * 2^(p+1)) < 2^15.
    """
    key = (n_in, n_out, str(device))
    if key not in _COEFFS:
        scale = n_in / n_out
        fs = max(scale, 1.0)
        support = 3.0 * fs
        centers = (np.arange(n_out) + 0.5) * scale
        lo = np.maximum((centers - support + 0.5).astype(np.int64), 0)
        hi = np.minimum((centers + support + 0.5).astype(np.int64), n_in)
        idx = lo[:, None] + np.arange(int((hi - lo).max()))[None, :]
        valid = idx < hi[:, None]
        x = np.abs((idx - centers[:, None] + 0.5) / fs)
        w = np.where(valid & (x < 3.0), np.sinc(x) * np.sinc(x / 3.0), 0.0)
        w /= w.sum(1, keepdims=True)
        prec = 0
        for cur in range(_MAX_PREC):
            prec = cur
            if round(float(w.max()) * (1 << (cur + 1))) >= (1 << 15):
                break
        idx_t = torch.from_numpy(np.where(valid, idx, 0).T.copy()).to(device)
        w_t = torch.from_numpy(np.round(w * (1 << prec)).astype(np.int32).T.copy()).to(device)
        _COEFFS[key] = (idx_t, w_t, prec)
    return _COEFFS[key]


def _resample_axis(x: torch.Tensor, n_out: int, axis: int) -> torch.Tensor:
    """uint8 [H, W, C] -> uint8 with ``axis`` resampled to n_out (8-bit output, like the engine)."""
    n_in = x.shape[axis]
    if n_in == n_out:
        return x
    idx, wq, prec = _axis_coeffs(n_in, n_out, x.device)
    shape = [1, 1, 1]
    shape[axis] = n_out
    acc = torch.full(
        x.shape[:axis] + (n_out,) + x.shape[axis + 1 :],
        (1 << prec) >> 1,
        dtype=torch.int32,
        device=x.device,
    )
    for t in range(idx.shape[0]):
        acc += wq[t].view(shape) * x.index_select(axis, idx[t]).int()
    return (acc >> prec).clamp_(0, 255).to(torch.uint8)


def resize_lanczos3(img, h: int, w: int):
    """HWC uint8 (numpy or torch, any device) Lanczos3 resize, bit-exact with the engine.

    The engine's resizer (fast_image_resize) first box-reduces by an integer factor on
    axes shrunk by more than 2.5x (factor floor(scale), remainder rows / cols dropped,
    mean = sum * ceil(2^16 / n) >> 16), then convolves vertically, then horizontally,
    with i16 coefficients and 8-bit intermediates.
    """
    as_numpy = isinstance(img, np.ndarray)
    x = torch.from_numpy(img) if as_numpy else img
    H, W = x.shape[:2]
    ky = math.floor(H / h) if H / h > 2.5 else 1
    kx = math.floor(W / w) if W / w > 2.5 else 1
    if ky * kx > 1:
        Hr, Wr = H // ky, W // kx
        s = x[: Hr * ky, : Wr * kx].reshape(Hr, ky, Wr, kx, -1).sum((1, 3), dtype=torch.int32)
        x = ((s * -(-65536 // (ky * kx))) >> 16).to(torch.uint8)
    out = _resample_axis(_resample_axis(x, h, 0), w, 1)
    return out.numpy() if as_numpy else out


def overlap_crops(image, device=None) -> tuple[torch.Tensor, tuple[int, int]]:
    """HWC uint8 image (numpy or torch) -> crops uint8 [1 + th*tw, 378, 378, 3] on ``device``
    (global crop first) and the (rows, cols) tiling."""
    x = torch.as_tensor(image)
    if device is not None:
        x = x.to(device, non_blocking=True)
    oh, ow = x.shape[:2]
    tiling = select_tiling(
        max(1, oh - 2 * MARGIN_PX), max(1, ow - 2 * MARGIN_PX), WINDOW, MAX_CROPS
    )
    big = resize_lanczos3(x, tiling[0] * WINDOW + 2 * MARGIN_PX, tiling[1] * WINDOW + 2 * MARGIN_PX)
    crops = [resize_lanczos3(x, CROP, CROP)]
    for ty in range(tiling[0]):
        for tx in range(tiling[1]):
            y0, x0 = ty * WINDOW, tx * WINDOW
            crops.append(big[y0 : y0 + CROP, x0 : x0 + CROP])
    return torch.stack(crops), tiling


def reconstruct(local: torch.Tensor, tiling: tuple[int, int]) -> torch.Tensor:
    """Local crop features [th*tw, 27, 27, C] -> stitched grid [(19*th+8), (19*tw+8), C]."""
    th, tw = tiling
    win = GRID - 2 * MARGIN
    out = local.new_zeros(win * th + 2 * MARGIN, win * tw + 2 * MARGIN, local.shape[-1])
    for ty in range(th):
        for tx in range(tw):
            y0 = 0 if ty == 0 else MARGIN
            y1 = GRID if ty == th - 1 else GRID - MARGIN
            x0 = 0 if tx == 0 else MARGIN
            x1 = GRID if tx == tw - 1 else GRID - MARGIN
            dy, dx = ty * win + y0, tx * win + x0
            out[dy : dy + y1 - y0, dx : dx + x1 - x0] = local[ty * tw + tx, y0:y1, x0:x1]
    return out


class VisionEncoder:
    def __init__(self, path: str, device, dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        p = "vision_encoder.encoder.model.visual."
        with safe_open(path, "pt", device=str(device)) as f:
            g = lambda k: f.get_tensor(k).to(dtype)  # noqa: E731
            self.patch_emb = (g(p + "patch_embed.linear.weight"), g(p + "patch_embed.linear.bias"))
            self.pos_emb = g(p + "pos_embed")  # [1, 729, 1152]
            self.post_ln = (g(p + "norm.weight"), g(p + "norm.bias"))
            self.blocks = []
            for i in range(ENC_LAYERS):
                b = f"{p}blocks.{i}."
                self.blocks.append(
                    {
                        n: (g(b + n + ".weight"), g(b + n + ".bias"))
                        for n in ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.fc1", "mlp.fc2")
                    }
                )
            q = "vision_encoder.projection.mlp."
            self.proj = (
                (g(q + "fc1.weight"), g(q + "fc1.bias")),
                (g(q + "fc2.weight"), g(q + "fc2.bias")),
            )

    @torch.no_grad()
    def encode_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """uint8 NHWC crops (any device) -> post-LN patch features [n, 729, 1152]."""
        x = crops.to(self.device).permute(0, 3, 1, 2).to(self.dtype)
        x = x.div_(255.0).sub_(0.5).div_(0.5)
        n = x.shape[0]
        x = (
            x.reshape(n, 3, GRID, PATCH, GRID, PATCH)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(n, GRID * GRID, 3 * PATCH * PATCH)
        )
        x = F.linear(x, *self.patch_emb) + self.pos_emb
        for b in self.blocks:
            h = F.layer_norm(x, (ENC_DIM,), *b["norm1"], eps=1e-5)
            q, k, v = (
                F.linear(h, *b["attn.qkv"])
                .view(n, -1, 3, ENC_HEADS, ENC_DIM // ENC_HEADS)
                .permute(2, 0, 3, 1, 4)
            )
            o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(n, -1, ENC_DIM)
            x = x + F.linear(o, *b["attn.proj"])
            h = F.layer_norm(x, (ENC_DIM,), *b["norm2"], eps=1e-5)
            x = x + F.linear(F.gelu(F.linear(h, *b["mlp.fc1"]), approximate="tanh"), *b["mlp.fc2"])
        return F.layer_norm(x, (ENC_DIM,), *self.post_ln, eps=1e-5)

    @torch.no_grad()
    def project(self, feats: torch.Tensor, tiling: tuple[int, int]) -> torch.Tensor:
        """One image's encoder output [1 + th*tw, 729, 1152] -> LM image tokens [729, 2048]."""
        local = reconstruct(feats[1:].view(-1, GRID, GRID, ENC_DIM), tiling).permute(2, 0, 1)
        pooled = F.adaptive_avg_pool2d(local, (GRID, GRID)).permute(1, 2, 0).reshape(-1, ENC_DIM)
        x = torch.cat([feats[0], pooled], -1)
        (w1, b1), (w2, b2) = self.proj
        return F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2)

    @torch.no_grad()
    def __call__(self, images: list, max_crops_per_batch: int = 128) -> torch.Tensor:
        """Images (PIL / path / bytes / array) -> image tokens [N, 729, 2048].

        JPEG decoding runs in a thread pool; resizing, tiling and encoding run on the GPU.
        """
        with ThreadPoolExecutor(8) as pool:
            decoded = pool.map(lambda im: torch.from_numpy(load_rgb(im)).pin_memory(), images)
            prepared = [overlap_crops(a, self.device) for a in decoded]
        out = torch.empty(len(prepared), GRID * GRID, 2048, device=self.device, dtype=self.dtype)
        i = 0
        while i < len(prepared):
            j, n = i, 0  # pack whole images until the crop budget is reached
            while j < len(prepared) and (j == i or n + len(prepared[j][0]) <= max_crops_per_batch):
                n += len(prepared[j][0])
                j += 1
            feats = self.encode_crops(torch.cat([c for c, _ in prepared[i:j]]))
            o = 0
            for k in range(i, j):
                c, tiling = prepared[k]
                out[k] = self.project(feats[o : o + len(c)], tiling)
                o += len(c)
            i = j
        return out
