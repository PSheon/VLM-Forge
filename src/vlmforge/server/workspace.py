"""The UI's workspace: images, review items and their corrections in SQLite (GOALS §6, §7).

``workspace/forge.sqlite`` plus ``workspace/images/<sha256>.<ext>`` for uploads and captures.
Held-out / release images are referenced by path, never copied. Every correction records
the reviewer and the time; a reviewer claims an item for ``CLAIM_MINUTES`` so several people
can work the same queue.

An item is one (image, skill, prompt) with the model's prediction and, once reviewed, the
human label:

    status   pending -> approved | corrected | discarded      (claimed_by / claimed_at while open)
             auto = the judge accepted it and it is not in the spot-check sample: no human
    verdict  the judge's accept | fix (``judge`` holds its JSON); spot = 1 marks the sample
    pred     {"boxes": [[x0,y0,x1,y1]]} | {"points": [[x,y]]} | {"text": "..."}   normalised 0-1
    label    same shape as pred, the human's final answer (approved = pred)
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

CLAIM_MINUTES = 10
SKILLS = ("detect", "point", "query", "caption")
STATUSES = ("pending", "auto", "approved", "corrected", "discarded")
SPOT_FRAC, SPOT_MIN = 0.1, 20  # judge-accepted items a human still sees (per queue)
# columns added after the first schema; (table, column, type) applied with ALTER TABLE
MIGRATIONS = (
    ("items", "judge", "text"),
    ("items", "verdict", "text"),
    ("items", "spot", "integer not null default 0"),
    ("queues", "judge", "text not null default ''"),
)

SCHEMA = """
create table if not exists images (
    id text primary key,            -- sha256 of the file
    path text not null,
    width integer not null,
    height integer not null,
    source text not null,           -- upload | capture:<nvr> | heldout | <release>
    added_at text not null
);
create table if not exists queues (
    id integer primary key,
    name text unique not null,
    skill text not null,
    model text not null,            -- base | adapter path
    created_at text not null,
    note text default ''
);
create table if not exists items (
    id integer primary key,
    queue_id integer references queues(id),
    image_id text not null references images(id),
    skill text not null,
    prompt text not null,           -- object (detect/point) | question (query) | length (caption)
    pred text,                      -- json
    pred_model text,
    label text,                     -- json, set when reviewed
    status text not null default 'pending',
    reviewer text,
    reviewed_at text,
    claimed_by text,
    claimed_at real,
    origin text not null default 'queue'   -- queue | workbench
);
create index if not exists items_queue on items(queue_id, status);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "forge.sqlite"
        self._local = threading.local()  # one connection per thread (FastAPI's thread pool)
        self.db.executescript(SCHEMA)
        for table, col, typ in MIGRATIONS:
            have = {r["name"] for r in self.db.execute(f"pragma table_info({table})")}
            if col not in have:
                self.db.execute(f"alter table {table} add column {col} {typ}")
        self.db.commit()
        self.lock = threading.Lock()

    @property
    def db(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path, check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    def sources(self) -> list[dict]:
        rows = self.db.execute(
            "select source, count(*) n from images group by source order by source"
        ).fetchall()
        return [{"source": r["source"], "n": r["n"]} for r in rows]

    # ------------------------------------------------------------------ images
    def add_image_bytes(self, data: bytes, source: str = "upload", ext: str = "jpg") -> dict:
        sha = hashlib.sha256(data).hexdigest()
        path = self.images_dir / f"{sha}.{ext}"
        if not path.exists():
            path.write_bytes(data)
        return self.add_image_path(path, source, sha)

    def add_image_path(self, path: Path, source: str, sha: str | None = None) -> dict:
        """Register an image that stays where it is (release / capture file)."""
        path = Path(path)
        sha = sha or sha256_file(path)
        with self.lock:
            row = self.db.execute("select * from images where id = ?", (sha,)).fetchone()
            if row is None:
                w, h = Image.open(path).size
                self.db.execute(
                    "insert into images values (?, ?, ?, ?, ?, ?)",
                    (sha, str(path), w, h, source, now()),
                )
                self.db.commit()
                row = self.db.execute("select * from images where id = ?", (sha,)).fetchone()
        return dict(row)

    def image(self, image_id: str) -> dict | None:
        row = self.db.execute("select * from images where id = ?", (image_id,)).fetchone()
        return dict(row) if row else None

    def images(self, source: str | None = None, limit: int = 500) -> list[dict]:
        q, args = "select * from images", []
        if source:
            q, args = q + " where source like ?", [source + "%"]
        q += " order by added_at desc limit ?"
        return [dict(r) for r in self.db.execute(q, [*args, limit])]

    # ------------------------------------------------------------------ queues
    def create_queue(
        self, name: str, skill: str, model: str, note: str = "", judge: str = ""
    ) -> dict:
        """``judge``: the judge model's name, or '' when every item goes to a human."""
        if skill not in SKILLS:
            raise ValueError(f"unknown skill {skill!r}")
        with self.lock:
            self.db.execute(
                "insert into queues (name, skill, model, created_at, note, judge)"
                " values (?, ?, ?, ?, ?, ?)",
                (name, skill, model, now(), note, judge),
            )
            self.db.commit()
        return self.queue_by_name(name)

    def queue_by_name(self, name: str) -> dict:
        return self._queue(
            self.db.execute("select * from queues where name = ?", (name,)).fetchone()
        )

    def queue(self, queue_id: int) -> dict:
        return self._queue(
            self.db.execute("select * from queues where id = ?", (queue_id,)).fetchone()
        )

    def _queue(self, row) -> dict:
        if row is None:
            raise KeyError("no such queue")
        q = dict(row)
        counts = self.db.execute(
            "select status, count(*) n from items where queue_id = ? group by status", (q["id"],)
        ).fetchall()
        q["counts"] = {s: 0 for s in STATUSES} | {r["status"]: r["n"] for r in counts}
        q["total"] = sum(q["counts"].values())
        verdicts = self.db.execute(
            "select coalesce(verdict, 'none') v, count(*) n from items where queue_id = ?"
            " group by v",
            (q["id"],),
        ).fetchall()
        q["verdicts"] = {r["v"]: r["n"] for r in verdicts}
        q["spot"] = self.db.execute(
            "select count(*) from items where queue_id = ? and spot = 1", (q["id"],)
        ).fetchone()[0]
        return q

    def queues(self) -> list[dict]:
        return [self._queue(r) for r in self.db.execute("select * from queues order by id desc")]

    # ------------------------------------------------------------------ items
    def add_item(
        self,
        image_id: str,
        skill: str,
        prompt: str,
        pred: dict | None,
        pred_model: str | None,
        queue_id: int | None = None,
        origin: str = "queue",
    ) -> int:
        with self.lock:
            cur = self.db.execute(
                "insert into items (queue_id, image_id, skill, prompt, pred, pred_model, origin)"
                " values (?, ?, ?, ?, ?, ?, ?)",
                (
                    queue_id,
                    image_id,
                    skill,
                    prompt,
                    json.dumps(pred) if pred is not None else None,
                    pred_model,
                    origin,
                ),
            )
            self.db.commit()
            return int(cur.lastrowid)

    def set_pred(self, item_id: int, pred: dict | None, pred_model: str | None) -> None:
        with self.lock:
            self.db.execute(
                "update items set pred = ?, pred_model = ? where id = ?",
                (json.dumps(pred) if pred is not None else None, pred_model, item_id),
            )
            self.db.commit()

    def set_judge(self, item_id: int, judge: dict) -> None:
        with self.lock:
            self.db.execute(
                "update items set judge = ?, verdict = ? where id = ?",
                (json.dumps(judge, ensure_ascii=False), judge.get("verdict"), item_id),
            )
            self.db.commit()

    def unjudged(self, queue_id: int) -> list[dict]:
        return [
            it
            for it in self.items(queue_id, "pending")
            if it["pred"] is not None and not it["verdict"]
        ]

    def triage(self, queue_id: int, seed: int = 0) -> dict:
        """After judging: accepted items leave the human queue (status ``auto``) except a
        random spot-check sample of max(SPOT_MIN, SPOT_FRAC) that stays ``pending`` with
        ``spot = 1``; ``fix`` items stay pending. Items a human already touched are left alone."""
        accepted = [it["id"] for it in self.items(queue_id, "pending") if it["verdict"] == "accept"]
        rng = random.Random(seed)
        rng.shuffle(accepted)
        k = min(len(accepted), max(SPOT_MIN, round(SPOT_FRAC * len(accepted))))
        spot, auto = accepted[:k], accepted[k:]
        with self.lock:
            if spot:
                self.db.execute(
                    f"update items set spot = 1 where id in ({','.join('?' * len(spot))})", spot
                )
            if auto:
                self.db.execute(
                    f"update items set status = 'auto' where id in ({','.join('?' * len(auto))})",
                    auto,
                )
            self.db.commit()
        return {"accepted": len(accepted), "spot": len(spot), "auto": len(auto)}

    def item(self, item_id: int) -> dict | None:
        row = self.db.execute(
            "select items.*, images.path, images.width, images.height, images.source"
            " from items join images on images.id = items.image_id where items.id = ?",
            (item_id,),
        ).fetchone()
        return self._item(row) if row else None

    def _item(self, row) -> dict:
        it = dict(row)
        it["pred"] = json.loads(it["pred"]) if it["pred"] else None
        it["label"] = json.loads(it["label"]) if it["label"] else None
        it["judge"] = json.loads(it["judge"]) if it.get("judge") else None
        return it

    def items(self, queue_id: int, status: str | None = None, limit: int = 2000) -> list[dict]:
        q = (
            "select items.*, images.path, images.width, images.height, images.source"
            " from items join images on images.id = items.image_id where queue_id = ?"
        )
        args: list = [queue_id]
        if status:
            q += " and status = ?"
            args.append(status)
        q += " order by items.id limit ?"
        return [self._item(r) for r in self.db.execute(q, [*args, limit])]

    def claim_next(self, queue_id: int, reviewer: str, after: int | None = None) -> dict | None:
        """The next pending item nobody else holds (claims expire after CLAIM_MINUTES): the
        judge's ``fix`` items first, then the spot-check sample, then unjudged ones."""
        with self.lock:
            cutoff = time.time() - CLAIM_MINUTES * 60
            q = (
                "select id from items where queue_id = ? and status = 'pending'"
                " and (claimed_by is null or claimed_by = ? or claimed_at < ?)"
            )
            args: list = [queue_id, reviewer, cutoff]
            if after is not None:
                q += " and id > ?"
                args.append(after)
            order = (
                " order by case when verdict = 'fix' then 0 when spot = 1 then 1 else 2 end,"
                " id limit 1"
            )
            row = self.db.execute(q + order, args).fetchone()
            if row is None:
                return None
            self.db.execute(
                "update items set claimed_by = ?, claimed_at = ? where id = ?",
                (reviewer, time.time(), row["id"]),
            )
            self.db.commit()
        return self.item(row["id"])

    def review(self, item_id: int, action: str, reviewer: str, label: dict | None = None) -> dict:
        """approve (label = pred), correct (label given) or discard an item."""
        it = self.item(item_id)
        if it is None:
            raise KeyError("no such item")
        if action == "approve":
            final = it["pred"]
        elif action == "correct":
            if label is None:
                raise ValueError("correct needs a label")
            final = label
        elif action == "discard":
            final = None
        else:
            raise ValueError(f"unknown action {action!r}")
        status = {"approve": "approved", "correct": "corrected", "discard": "discarded"}[action]
        with self.lock:
            self.db.execute(
                "update items set label = ?, status = ?, reviewer = ?, reviewed_at = ?,"
                " claimed_by = null, claimed_at = null where id = ?",
                (
                    json.dumps(final) if final is not None else None,
                    status,
                    reviewer,
                    now(),
                    item_id,
                ),
            )
            self.db.commit()
        return self.item(item_id)

    def reopen(self, item_id: int) -> dict:
        with self.lock:
            self.db.execute(
                "update items set status = 'pending', label = null, reviewer = null,"
                " reviewed_at = null where id = ?",
                (item_id,),
            )
            self.db.commit()
        return self.item(item_id)

    # ------------------------------------------------------------------ export
    def reviewed_rows(self, queue_ids: list[int] | None = None) -> list[dict]:
        """Training / eval rows (``vlmforge.data`` format): human-reviewed items (``human``)
        and judge-accepted ones (``machine_reviewed``)."""
        q = (
            "select items.*, images.path, images.width, images.height, images.source"
            " from items join images on images.id = items.image_id"
            " where status in ('approved', 'corrected', 'auto')"
        )
        args: list = []
        if queue_ids:
            q += f" and queue_id in ({','.join('?' * len(queue_ids))})"
            args += queue_ids
        rows = []
        for r in self.db.execute(q + " order by items.id", args):
            it = self._item(r)
            rows.append(item_row(it))
        return rows


def item_row(it: dict) -> dict:
    """One reviewed item -> a training / eval row. A human's verdict is ``human``; a
    judge-accepted item nobody looked at is ``machine_reviewed`` (rules §6)."""
    auto = it["status"] == "auto"
    judge = it.get("judge") or {}
    base = {
        "image": it["path"],
        "source": it["source"],
        "id": f"forge/{it['id']}",
        "skill": it["skill"],
        "label_origin": "machine_reviewed" if auto else "human",
        "reviewer": f"judge:{judge.get('model', '?')}" if auto else it["reviewer"],
        "date": (it["reviewed_at"] or "")[:10],
        "prefill": it["pred_model"],
        "changed": it["status"] == "corrected",
        "judge": it.get("verdict"),
    }
    lab = (it["pred"] if auto else it["label"]) or {}
    if it["skill"] == "detect":
        return {**base, "object": it["prompt"], "boxes": lab.get("boxes", [])}
    if it["skill"] == "point":
        return {**base, "object": it["prompt"], "points": lab.get("points", [])}
    if it["skill"] == "query":
        return {**base, "question": it["prompt"], "answer": lab.get("text", "")}
    return {**base, "length": it["prompt"], "answer": lab.get("text", "")}
