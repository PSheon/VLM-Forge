import json

from vlmforge.autogen import parse_teacher, repair_json

MALFORMED = """{
  "objects": [
    {"name": "machine", "boxes": [[198, 48, 465, 495]]},
    {"name": "shelf", "boxes": [0, 0, 300, 355]},
    {"name": "box", "boxes": [54, 125, 120, 208], [38, 283, 105, 462], [100, 250, 145, 345]},
    {"name": "stool", "boxes": [425, 300, 465, 412], [750, 125, 770, 195]}
  ],
  "questions": [{"q": "Is any staff present?", "a": "No"}],
  "caption": "An empty factory floor."
}"""

TRUNCATED = """{
  "objects": [
    {"name": "table", "boxes": [[175, 525, 323, 812], [71, 431, 185, 600]]},
    {"name": "chair", "boxes": [[305, 478, 390, 690], [137, 755, 265, 890]]},
    {"name": "locker", "boxes": [[665, 245, 755, 735], [585, 125"""


def test_repair_unwrapped_boxes():
    t = parse_teacher(MALFORMED)
    assert [(o["name"], len(o["boxes"])) for o in t["objects"]] == [
        ("machine", 1),
        ("shelf", 1),
        ("box", 3),
        ("stool", 2),
    ]
    assert t["objects"][1]["boxes"][0] == [0.0, 0.0, 0.3, 0.355]
    assert t["questions"][0]["a"] == "No" and t["caption"]


def test_repair_truncated_output_keeps_complete_objects():
    fixed = repair_json(TRUNCATED)
    json.loads(fixed)  # valid again
    t = parse_teacher(TRUNCATED)
    assert [o["name"] for o in t["objects"]] == ["table", "chair"]
    assert t["questions"] == [] and t["caption"] == ""


def test_repair_leaves_good_json_alone():
    good = (
        '{"objects": [{"name": "a", "boxes": [[1, 2, 3, 4], [5, 6, 7, 8]]}],'
        ' "questions": [], "caption": ""}'
    )
    assert json.loads(repair_json(good)) == json.loads(good)
