import json

import pytest

from vlmforge.evaluate import macro, match_boxes, match_points_in_boxes, number, score
from vlmforge.gate import combined_macro, gate


def test_matchers_and_number():
    assert match_boxes([[0, 0, 1, 1]], [[0, 0, 1, 1]]) == (1, 0, 0)
    assert match_boxes([[0, 0, 0.1, 0.1]], [[0.5, 0.5, 1, 1]]) == (0, 1, 1)
    assert match_points_in_boxes([[0.5, 0.5], [0.9, 0.9]], [[0, 0, 1, 1]]) == (1, 1, 0)
    assert number("There are three people") == 3 and number("none") is None


def test_score_macro_groups_by_source_and_object():
    rows = [
        {
            "image": "a",
            "source": "cctv",
            "skill": "detect",
            "object": "person",
            "boxes": [[0, 0, 1, 1]],
        },
        {"image": "b", "source": "cctv", "skill": "detect", "object": "shelf", "boxes": []},
        {"image": "a", "source": "cctv", "skill": "query", "question": "How many?", "answer": "1"},
    ]
    preds = [[[0, 0, 1, 1]], [[0, 0, 0.5, 0.5]], "There is 1 person."]
    m = score(rows, preds)["metrics"]
    assert m["detect@cctv:person/f1"] == 1.0
    assert m["detect@cctv:shelf/f1"] == 0.0  # negative item with a false positive
    assert m["detect/macro"] == 0.5 and m["query/macro"] == 1.0
    assert macro(m) == {"detect": 0.5, "query": 1.0}


def test_gate_rules(tmp_path):
    base = {"detect": 0.40, "point": 0.50, "query": 0.90}
    assert gate(base, {"detect": 0.45, "point": 0.495, "query": 0.90}, "detect")["passed"]
    assert not gate(base, {"detect": 0.39, "point": 0.50, "query": 0.90}, "detect")["passed"]
    assert not gate(base, {"detect": 0.45, "point": 0.48, "query": 0.90}, "detect")["passed"]
    assert gate(base, {"detect": 0.41, "point": 0.495, "query": 0.895}, "multi")["passed"]
    assert not gate(base, {"detect": 0.40, "point": 0.50, "query": 0.90}, "multi")["passed"]
    with pytest.raises(ValueError):
        gate(base, {"detect": 0.5}, "caption")
    # combined over suites = mean of each suite's macro
    for i, v in enumerate((0.2, 0.4)):
        (tmp_path / f"r{i}.json").write_text(json.dumps({"metrics": {"detect/macro": v}}))
    assert combined_macro(sorted(tmp_path.glob("r*.json"))) == {"detect": pytest.approx(0.3)}
