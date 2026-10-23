from PIL import Image

from vlmforge.judge import parse_json, prompt_for, render, verdict_from


def test_parse_and_verdicts():
    assert parse_json('Sure: {"wrong": [2], "missed": 0, "note": "box 2 is a chair"}') == {
        "wrong": [2],
        "missed": 0,
        "note": "box 2 is a chair",
    }
    assert verdict_from("detect", {"wrong": [], "missed": 0})["verdict"] == "accept"
    assert verdict_from("detect", {"wrong": [2], "missed": 0})["verdict"] == "fix"
    assert verdict_from("detect", {"wrong": [], "missed": "1"})["verdict"] == "fix"
    assert verdict_from("point", {})["verdict"] == "fix"  # unparseable -> a human looks
    assert verdict_from("query", {"correct": True, "answer": "3"})["verdict"] == "accept"
    assert verdict_from("query", {"correct": "true"})["verdict"] == "fix"
    assert verdict_from("caption", {"unsupported": []})["verdict"] == "accept"
    assert verdict_from("caption", {"unsupported": ["a red car"]})["verdict"] == "fix"


def test_prompts_and_render():
    assert "2 red boxes" in prompt_for("detect", "person", {"boxes": [[0, 0, 1, 1], [0, 0, 1, 1]]})
    assert 'no "person"' in prompt_for("detect", "person", {"boxes": []})
    assert "How many?" in prompt_for("query", "How many?", {"text": "2"})
    im = Image.new("RGB", (3000, 1500), "gray")
    out = render(im, {"boxes": [[0.1, 0.1, 0.5, 0.5]], "points": [[0.7, 0.7]]})
    assert out.width == 1280 and out.height == 640
    assert out.getpixel((int(0.1 * 1280) + 1, int(0.3 * 640))) != (128, 128, 128)  # box edge
