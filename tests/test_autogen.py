from vlmforge.autogen import agree, auto_label, items_from_teacher, parse_teacher, teacher_prompt

TEACHER = """{"objects": [{"name": "Person", "boxes": [[100, 200, 300, 900], [500, 100, 600, 400]]},
 {"name": "forklift", "boxes": []}, {"bad": 1}],
 "questions": [{"q": "How many people are there?", "a": "2"},
  {"q": "Is anyone wearing a helmet?", "a": "No"}, {"q": "", "a": "x"}],
 "caption": "Two people stand in a warehouse aisle."}"""


def test_parse_and_items():
    t = parse_teacher(TEACHER)
    assert [o["name"] for o in t["objects"]] == ["person", "forklift"]
    assert t["objects"][0]["boxes"][0] == [0.1, 0.2, 0.3, 0.9]
    assert len(t["questions"]) == 2 and t["caption"].startswith("Two people")
    items = items_from_teacher(t, {"detect", "point", "query", "caption"})
    kinds = [(i["skill"], i["prompt"]) for i in items]
    assert kinds == [
        ("detect", "person"),
        ("point", "person"),
        ("detect", "forklift"),  # no boxes -> a negative detect item, no point item
        ("query", "How many people are there?"),
        ("query", "Is anyone wearing a helmet?"),
        ("caption", "normal"),
    ]
    assert items[1]["teacher"]["points"][0] == [0.2, 0.55]
    assert "safety helmet" in teacher_prompt() and "spill" not in teacher_prompt()
    assert "ladders" in teacher_prompt(topics=["whether ladders are secured"])


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
