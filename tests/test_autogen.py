from vlmforge.autogen import (
    agree,
    auto_label,
    ground_prompt,
    items_from_teacher,
    parse_boxes,
    parse_teacher,
    teacher_prompt,
)

TEACHER = """{"objects": [{"name": "Person", "count": 2}, {"name": "forklift", "count": 0},
 {"name": "person", "count": 9}, {"bad": 1}],
 "questions": [{"q": "How many people are there?", "a": "2"},
  {"q": "Is anyone wearing a helmet?", "a": "No"}, {"q": "", "a": "x"}],
 "caption": "Two people stand in a warehouse aisle."}"""


def test_parse_and_items():
    t = parse_teacher(TEACHER)
    assert [(o["name"], o["count"]) for o in t["objects"]] == [("person", 2), ("forklift", 0)]
    assert len(t["questions"]) == 2 and t["caption"].startswith("Two people")
    t["objects"][0]["boxes"] = [[0.1, 0.2, 0.3, 0.9], [0.5, 0.1, 0.6, 0.4]]
    t["objects"].append({"name": "chair", "count": 30, "boxes": [], "skipped": True})
    items = items_from_teacher(t, {"detect", "point", "query", "caption"})
    kinds = [(i["skill"], i["prompt"]) for i in items]
    assert kinds == [
        ("detect", "person"),
        ("point", "person"),
        ("detect", "forklift"),  # no boxes -> a negative detect item, no point item
        ("query", "How many people are there?"),
        ("query", "Is anyone wearing a helmet?"),
        ("caption", "normal"),
    ]  # the skipped chair type produces nothing
    assert items[1]["teacher"]["points"][0] == [0.2, 0.55]
    assert "safety helmet" in teacher_prompt() and "spill" not in teacher_prompt()
    assert "ladders" in teacher_prompt(topics=["whether ladders are secured"])
    assert '"person" in this image (about 2)' in ground_prompt("person", 2)


def test_parse_boxes_tolerates_loops_and_cuts():
    assert parse_boxes("[[100, 200, 300, 900], [500, 100, 600, 400]]") == [
        [0.1, 0.2, 0.3, 0.9],
        [0.5, 0.1, 0.6, 0.4],
    ]
    assert parse_boxes("[100, 200, 300, 900]") == [[0.1, 0.2, 0.3, 0.9]]
    assert parse_boxes("[]") == [] and parse_boxes("none") == []
    looped = "[" + ", ".join(["[15, 0, 55, 62]"] * 40) + ", [15, 0, 55"
    assert parse_boxes(looped) == [[0.015, 0.0, 0.055, 0.062]]  # dedup + truncated tail
    many = "[" + ", ".join(f"[{i * 10}, 0, {i * 10 + 5}, 10]" for i in range(30)) + "]"
    assert len(parse_boxes(many)) == 12


def test_agreement_rules():
    t = {"boxes": [[0.1, 0.1, 0.5, 0.5]]}
    assert agree("detect", {"boxes": [[0.1, 0.1, 0.5, 0.5]]}, t)["verdict"] == "accept"
    assert agree("detect", {"boxes": []}, t)["verdict"] == "fix"
    assert (
        agree("detect", {"boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]}, t)["verdict"]
        == "fix"
    )
    assert agree("point", {"points": [[0.3, 0.3]]}, t)["verdict"] == "accept"
    assert agree("point", {"points": [[0.8, 0.8]]}, t)["verdict"] == "fix"
    assert agree("query", {"text": "There are 2 people."}, {"text": "2"})["verdict"] == "accept"
    assert agree("query", {"text": "three"}, {"text": "2"})["verdict"] == "fix"
    assert agree("query", {"text": "No, nobody wears one."}, {"text": "No"})["verdict"] == "accept"
    assert agree("query", {"text": "Yes"}, {"text": "No"})["verdict"] == "fix"
    assert (
        agree("query", {"text": "The vest is orange."}, {"text": "orange"})["verdict"] == "accept"
    )
    assert agree("query", None, {"text": "2"})["verdict"] == "fix"
    assert agree("caption", {"text": "x"}, {"text": "y"})["verdict"] == "fix"


def test_auto_label_prefers_teacher_text_and_student_geometry():
    assert auto_label("query", {"text": "2 people"}, {"text": "2"}) == {"text": "2"}
    assert auto_label("detect", {"boxes": [[0, 0, 1, 1]]}, {"boxes": [[0, 0, 0.9, 0.9]]}) == {
        "boxes": [[0, 0, 1, 1]]
    }
    assert auto_label("detect", None, {"boxes": [[0, 0, 0.9, 0.9]]}) == {
        "boxes": [[0, 0, 0.9, 0.9]]
    }


def test_grounded_count_answer():
    """MD3 answers "How many forklifts?" with "There is <ground> x y <eos>": the text alone is
    "There is ", the grounded point is the count (seen on 4 of 1 313 warehouse questions)."""
    from vlmforge.model.moondream import COORD_ID, EOS, Generation, grounded_answer

    g = Generation(
        tokens=[2272, 41649, 7, COORD_ID, COORD_ID, EOS],
        values=[(0, 0), (0, 0), (0, 0), (0.41, 0), (0.62, 0), (0, 0)],
    )
    a = grounded_answer(
        g,
        lambda ids: (
            {2272: "There", 41649: " is ", 7: ""}.get(ids[0], "") if len(ids) == 1 else "There is "
        ),
    )
    assert a == {"text": "There is ", "points": [[0.41, 0.62]]}
    one = agree("query", {"text": a["text"], "points": a["points"]}, {"text": "1"})
    assert one["verdict"] == "accept" and "指了 1 個點" in one["note"]
    assert agree("query", {"text": "There is "}, {"text": "1"})["verdict"] == "fix"
    assert (
        agree("query", {"text": "There are 2 forklifts.", "points": [[0.1, 0.1]]}, {"text": "2"})[
            "verdict"
        ]
        == "accept"
    )


def test_probes_add_negatives_for_what_the_teacher_did_not_list():
    """Auto mode asks the probe objects on every image; absent ones become 0-box detect
    questions, unless the teacher listed them under another name (worker = person)."""
    from vlmforge.autogen import DEFAULT_PROBES, items_from_teacher

    t = {
        "objects": [
            {"name": "worker", "count": 2, "boxes": [[0.1, 0.1, 0.2, 0.3]]},
            {"name": "pallet", "count": 1, "boxes": []},
        ],
        "questions": [],
        "caption": "",
    }
    items = items_from_teacher(t, {"detect", "point"}, DEFAULT_PROBES)
    zero = sorted(
        i["prompt"] for i in items if i["skill"] == "detect" and not i["teacher"]["boxes"]
    )
    assert zero == ["forklift", "helmet", "pallet"]  # two probes + the listed-but-unboxed pallet
    assert not any(i["prompt"] == "person" for i in items)  # listed as "worker"
    assert items_from_teacher(t, {"query"}, DEFAULT_PROBES) == []  # no detect skill, no probes
    assert not [i for i in items_from_teacher(t, {"detect"}) if i["prompt"] in DEFAULT_PROBES]
