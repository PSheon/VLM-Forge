"""Moondream 3.1 in plain PyTorch (vision encoder, text model, region model, LoRA).

Vendored from VisionTraining's parity-verified teacher port so it runs on the GB300, where
the upstream engine does not (GOALS §3).

    from vlmforge.model import Moondream, load_adapter
    m = Moondream(weights, tokenizer_dir, **load_adapter("runs/x/x.pt", "cuda"))
    m.detect(images, "person")

Adapter file (``.pt``): ``{"lora_state_dict", "rank", "alpha", "region_head"?}``. The LoRA
part is the engine's TextLoRA layout (scale folded in); ``region_head`` holds the fine-tuned
region decoder under upstream tensor names and is applied by ``RegionModel.load_head``.
"""

from pathlib import Path

import torch

from .moondream import Generation, Moondream, Request, boxes_from, points_from, prompt_ids
from .text import KVCache, RegionHead, RegionModel, TextLoRA, TextModel
from .vision import VisionEncoder


def load_adapter(path: str | Path, device) -> dict:
    """``Moondream(...)`` keyword arguments for an adapter file: ``lora`` and ``region_head``."""
    ckpt = torch.load(path, map_location="cpu")
    lora = TextLoRA.from_checkpoint(str(path), device)
    return {"lora": lora, "region_head": ckpt.get("region_head")}


__all__ = [
    "Moondream",
    "Generation",
    "Request",
    "boxes_from",
    "points_from",
    "prompt_ids",
    "load_adapter",
    "TextModel",
    "RegionModel",
    "RegionHead",
    "TextLoRA",
    "KVCache",
    "VisionEncoder",
]
