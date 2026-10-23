import json

import torch

from vlmforge.model.moondream import COORD_ID, EOS, IMAGE_TOKENS, SIZE_ID
from vlmforge.model.text import RegionModel, TextLoRA
from vlmforge.train import (
    bf16_view,
    gt_answer,
    layout,
    load_rows,
    lr_at,
    replay_prompts,
)


class FakeTok:  # detect / point prompts only need the object's ids
    def encode(self, s):
        return type("Encoding", (), {"ids": [1000]})()


def spatial(row, max_objects=50):
    return gt_answer(FakeTok(), row, max_objects, 384)


def test_detect_answer_left_to_right_and_layout():
    row = {
        "skill": "detect",
        "object": "person",
        "boxes": [[0.5, 0.5, 0.7, 0.9], [0.1, 0.2, 0.3, 0.4]],
    }
    p, ans = spatial(row)
    assert p[-1] == 3  # <answer>
    assert [t for t, _ in ans] == [COORD_ID, COORD_ID, SIZE_ID, COORD_ID, COORD_ID, SIZE_ID, EOS]
    assert abs(ans[0][1] - 0.2) < 1e-6 and abs(ans[3][1] - 0.6) < 1e-6
    n, tok, coord, size = layout(len(p), ans)
    start = IMAGE_TOKENS + len(p) - 1  # the <answer> token predicts the first coordinate
    assert tok[0] == (start, COORD_ID) and tok[-1] == (start + 6, EOS)
    assert [c[0] for c in coord] == [start, start + 1, start + 3, start + 4]
    assert coord[0][1] == int(RegionModel.coord_bin(torch.tensor(0.2)))
    assert [s[0] for s in size] == [start + 2, start + 5]
    assert n == start + len(ans)  # BOS + image + prompt[1:] + answer[:-1]


def test_truncated_objects_have_no_eos():
    row = {"skill": "point", "object": "x", "points": [[i / 10, 0.5] for i in range(5)]}
    ans = spatial(row, max_objects=3)[1]
    assert len(ans) == 6 and EOS not in [t for t, _ in ans]
    assert [t for t, _ in spatial(row)[1]][-1] == EOS


def test_empty_detect_is_just_eos():
    assert spatial({"skill": "detect", "object": "x", "boxes": []})[1] == [(EOS, None)]


def test_replay_skips_trained_objects():
    prompts = replay_prompts(
        {"questions": ["q?"], "captions": [], "objects": ["Person", "tree"]}, {"person"}
    )
    assert prompts["text"] == [("query", {"question": "q?"})]
    assert prompts["spatial"] == [("detect", {"object": "tree"}), ("point", {"object": "tree"})]


def test_load_rows_drops_eval_images(tmp_path):
    train = tmp_path / "train.jsonl"
    ev = tmp_path / "eval.jsonl"
    rows = [
        {
            "image": f"/r/train/data/files/s/{i}.jpg",
            "skill": "query",
            "question": "q",
            "answer": "a",
        }
        for i in range(3)
    ]
    train.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ev.write_text(json.dumps({"image": "/r/eval/data/files/s/1.jpg"}) + "\n")
    kept, stats = load_rows({"data": {"train": [str(train)], "exclude": [str(ev)]}})
    assert len(kept) == 2 and stats["dropped_eval_images"] == 1


def test_lora_alpha_scale_is_folded_into_b():
    lora = TextLoRA(8, "cpu", dtype=torch.float32, alpha=16)
    assert lora.scale == 2.0
    with torch.no_grad():
        lora.dense[0].up_b.fill_(1.0)
    view = bf16_view(lora).dense[0]
    assert float(view.up_b[0, 0].detach()) == 2.0
    assert float(view.up_a[0, 0].detach()) == float(lora.dense[0].up_a[0, 0].detach().bfloat16())
    sd = lora.engine_state_dict()
    assert float(sd["text_lora.dense.0.up_b"][0, 0]) == 2.0
    assert torch.equal(sd["text_lora.dense.0.up_a"], lora.dense[0].up_a)


def test_lr_schedule():
    o = {"lr": 1.0, "warmup": 2, "steps": 10, "min_lr_ratio": 0.1}
    assert lr_at(0, o) == 0.5 and lr_at(1, o) == 1.0
    assert abs(lr_at(10, o) - 0.1) < 1e-9
