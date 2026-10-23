"""The Orin bundle: its loader is valid Python and our adapter keys are the ones kestrel's
``LoRA(text=TextLoRA)`` registers (``text_lora.dense.*`` / ``text_lora.moe.*``)."""

import ast
import re

from vlmforge.orin import BUNDLE_FILES, MODEL_ID, local_adapter_source


def test_local_adapter_source_parses():
    src = local_adapter_source()
    tree = ast.parse(src)
    names = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    assert {"LocalAdapterProvider", "Tuned"} <= names
    assert f'MODEL_ID = "{MODEL_ID}"' in src
    assert "local_adapter.py" in BUNDLE_FILES


def test_engine_keys_match_kestrel_layout():
    import torch

    from vlmforge.model.text import TextLoRA

    sd = TextLoRA(8, torch.device("cpu"), alpha=16).engine_state_dict()
    pat = re.compile(r"^text_lora\.(dense|moe)\.\d+\.(up|down)_(a|b)$")
    assert all(pat.match(k) for k in sd), [k for k in sd if not pat.match(k)][:3]
    assert sd["text_lora.dense.0.up_a"].shape[0] == 8  # kestrel reads the rank from here
    assert sd["text_lora.moe.0.up_a"].shape[1] == 1  # rank / experts_per_token (8)
