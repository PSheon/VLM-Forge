"""The UI's workspace: images, review items and their corrections in SQLite (GOALS §4, §5).

``workspace/forge.sqlite`` plus ``workspace/images/<sha256>.<ext>`` for uploads.
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
import os
import random
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageOps

CLAIM_MINUTES = 10
SKILLS = ("detect", "point", "query", "caption")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STATUSES = ("pending", "auto", "approved", "corrected", "discarded")
BACKUP_KEEP = 7  # daily database copies kept in workspace/backups
SPOT_FRAC, SPOT_MIN = 0.1, 20  # judge-accepted items a human still sees (per queue)
ALL = 10**9  # ``items(limit=ALL)``: every row, for pipelines
# columns added after the first schema; (table, column, type) applied with ALTER TABLE
MIGRATIONS = (
    ("items", "judge", "text"),
    ("items", "verdict", "text"),
    ("items", "spot", "integer not null default 0"),
    ("queues", "judge", "text not null default ''"),
    ("items", "teacher", "text"),  # autogen: the teacher's answer (json)
    ("queues", "mode", "text not null default 'manual'"),  # manual | auto
    ("queues", "config", "text"),  # autogen config (json)
)

SCHEMA = """
create table if not exists images (
    id text primary key,            -- sha256 of the file
    path text not null,
    width integer not null,
    height integer not null,
    source text not null,           -- upload | folder:<name> | heldout
    added_at text not null
);
create table if not exists image_sigs (  -- near-duplicate check (datasets.DUP_P99)
    image_id text primary key,
    sig blob not null               -- 64x36 RGB, row-major uint8
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


def signature(im: Image.Image) -> bytes:
    """A 64x36 colour thumbnail: enough to tell 'the same frame again' from 'something in it
    changed' (datasets.DUP_P99; colour, so a lamp turning red to green counts), small enough
    to compare thousands at once."""
    im = ImageOps.exif_transpose(im).convert("RGB").resize((64, 36), Image.BILINEAR)
    return im.tobytes()


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
        self.daily: list = []  # chores run with the backup check (e.g. DatasetStore.purge)

    @property
    def db(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path, check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    # ------------------------------------------------------------------ backups
    def backup(self, keep: int = BACKUP_KEEP) -> Path:
        """Today's copy of the database (every question, answer and review) in
        ``backups/forge-YYYYMMDD.sqlite``, made once a day with SQLite's online backup; the
        newest ``keep`` stay. Image files are not copied (uploads live in ``images/``)."""
        d = self.root / "backups"
        d.mkdir(exist_ok=True)
        dest = d / f"forge-{time.strftime('%Y%m%d')}.sqlite"
        if not dest.exists():
            tmp = dest.with_suffix(".tmp")
            src, dst = sqlite3.connect(self._path), sqlite3.connect(tmp)
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            os.replace(tmp, dest)
        for old in sorted(d.glob("forge-*.sqlite"))[:-keep]:
            old.unlink()
        return dest

    def latest_backup(self) -> dict | None:
        files = sorted((self.root / "backups").glob("forge-*.sqlite"))
        if not files:
            return None
        st = files[-1].stat()
        return {
            "file": str(files[-1]),
            "at": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="minutes"),
            "count": len(files),
        }

    def start_backups(self, every_s: int = 3600) -> None:
        """Back up now and check hourly (one copy per day) on a daemon thread; ``daily``
        chores run with each check."""

        def loop():
            while True:
                for job in (self.backup, *self.daily):
                    try:
                        job()
                    except Exception as e:  # noqa: BLE001 - a failed chore must not stop the server
                        print(f"{getattr(job, '__name__', job)} failed: {e}", flush=True)
                time.sleep(every_s)

        threading.Thread(target=loop, daemon=True, name="backups").start()

    def sources(self) -> list[dict]:
        rows = self.db.execute(
            "select source, count(*) n from images group by source order by source"
        ).fetchall()
        return [{"source": r["source"], "n": r["n"]} for r in rows]

    # ------------------------------------------------------------------ images
    def signatures(self, image_ids: list[str]) -> dict[str, bytes]:
        """Each image's near-duplicate signature (``signature``), computed once and kept."""
        out: dict[str, bytes] = {}
        for k in range(0, len(image_ids), 500):
            chunk = image_ids[k : k + 500]
            marks = ",".join("?" * len(chunk))
            q = f"select image_id, sig from image_sigs where image_id in ({marks})"
            out.update({r[0]: r[1] for r in self.db.execute(q, chunk)})
        missing = [i for i in image_ids if i not in out]
        for i in missing:
            row = self.image(i)
            if row and Path(row["path"]).exists():
                out[i] = signature(Image.open(row["path"]))
                self.put_signature(i, out[i])
        return out

    def put_signature(self, image_id: str, sig: bytes) -> None:
        with self.lock:
            self.db.execute("insert or replace into image_sigs values (?, ?)", (image_id, sig))
            self.db.commit()

    def add_image_bytes(self, data: bytes, source: str = "upload", ext: str = "jpg") -> dict:
        sha = hashlib.sha256(data).hexdigest()
        path = self.images_dir / f"{sha}.{ext}"
        if not path.exists():
            path.write_bytes(data)
        return self.add_image_path(path, source, sha)

    def add_image_path(self, path: Path, source: str, sha: str | None = None) -> dict:
        """Register an image that stays where it is (a GB300 folder or a held-out file)."""
        path = Path(path)
        sha = sha or sha256_file(path)
        with self.lock:
            row = self.db.execute("select * from images where id = ?", (sha,)).fetchone()
            if row is None:
                w, h = Image.open(path).size
                self.db.execute(
                    "insert into images (id, path, width, height, source, added_at)"
                    " values (?, ?, ?, ?, ?, ?)",
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
        q["images_pending"] = self.db.execute(
            "select count(distinct image_id) from items where queue_id = ? and status = 'pending'",
            (q["id"],),
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
        teacher: dict | None = None,
    ) -> int:
        with self.lock:
            cur = self.db.execute(
                "insert into items (queue_id, image_id, skill, prompt, pred, pred_model, origin,"
                " teacher) values (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    queue_id,
                    image_id,
                    skill,
                    prompt,
                    json.dumps(pred) if pred is not None else None,
                    pred_model,
                    origin,
                    json.dumps(teacher, ensure_ascii=False) if teacher is not None else None,
                ),
            )
            self.db.commit()
            return int(cur.lastrowid)

    def set_queue_mode(self, queue_id: int, mode: str, config: dict) -> None:
        with self.lock:
            self.db.execute(
                "update queues set mode = ?, config = ? where id = ?",
                (mode, json.dumps(config, ensure_ascii=False), queue_id),
            )
            self.db.commit()

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

    def unfinished(self, queue_id: int, image_ids: list[str]) -> dict:
        """What an interrupted autogen batch still owes: images with no questions yet,
        pending items the student has not answered, pending items not yet compared. A
        server restart loses the in-memory job; ``/api/queues/{id}/resume`` uses this."""
        done = {
            r[0]
            for r in self.db.execute(
                "select distinct image_id from items where queue_id = ?", (queue_id,)
            )
        }
        pend = self.items(queue_id, "pending", ALL)
        return {
            "images": sum(i not in done for i in image_ids),
            "unpredicted": sum(it["pred"] is None for it in pend),
            "unjudged": sum(not it["verdict"] for it in pend),
        }

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
        it["teacher"] = json.loads(it["teacher"]) if it.get("teacher") else None
        return it

    def items(self, queue_id: int, status: str | None = None, limit: int = 2000) -> list[dict]:
        """Items of a queue (joined with the image row); pass ``limit=ALL`` from pipelines —
        the default is for the UI, and a 2 286-item batch once lost its tail to it."""
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

    # ------------------------------------------------------------------ per-image review
    def triage_items(self, queue_id: int, seed: int = 0) -> dict:
        """Autogen triage by question (GOALS §5: 一致→自動通過、不一致→人看): every pending
        item the student and teacher agreed on gets its auto label and leaves the human
        queue (status ``auto``), except a spot sample of max(SPOT_MIN, SPOT_FRAC) accepted
        items that stays pending with ``spot = 1``; ``fix`` and uncompared items stay pending.
        An image is shown to a human only while it still has pending items, so a 24-question
        image with two disagreements costs two looks, not 24."""
        from ..autogen import auto_label

        pend = self.items(queue_id, "pending", ALL)
        # idempotent: items already drawn for the spot check stay drawn (re-running triage on a
        # half-reviewed batch once auto-passed 128 of them), and the 10 % budget counts every
        # accepted item of the batch — auto-passed and already-sampled ones included
        accepted = [it for it in pend if it["verdict"] == "accept" and not it["spot"]]
        drawn = {it["id"] for it in pend if it["spot"]}
        n_acc = self.db.execute(
            "select count(*) from items where queue_id = ? and verdict = 'accept'", (queue_id,)
        ).fetchone()[0]
        n_spot = self.db.execute(
            "select count(*) from items where queue_id = ? and spot = 1", (queue_id,)
        ).fetchone()[0]
        rng = random.Random(seed)
        order = list(range(len(accepted)))
        rng.shuffle(order)
        k = min(len(accepted), max(0, max(SPOT_MIN, round(SPOT_FRAC * n_acc)) - n_spot))
        spot = {accepted[i]["id"] for i in order[:k]}
        n_auto = 0
        with self.lock:
            for it in accepted:
                if it["teacher"] is not None:
                    lab = auto_label(it["skill"], it["pred"], it["teacher"])
                    self.db.execute(
                        "update items set label = ? where id = ?",
                        (json.dumps(lab, ensure_ascii=False), it["id"]),
                    )
                if it["id"] in spot:
                    self.db.execute("update items set spot = 1 where id = ?", (it["id"],))
                else:
                    self.db.execute("update items set status = 'auto' where id = ?", (it["id"],))
                    n_auto += 1
            self.db.commit()
        spot |= drawn
        images = {it["image_id"] for it in pend}
        left = {it["image_id"] for it in pend if it["verdict"] != "accept" or it["id"] in spot}
        return {
            "items": len(pend),
            "accepted": len(accepted),
            "spot": len(spot),
            "auto": n_auto,
            "images": len(images),
            "images_left": len(left),
        }

    def claim_next_image(
        self,
        queue_id: int | None,
        reviewer: str,
        after_image: str | None = None,
        dataset_id: int | None = None,
        only_spot: bool = False,
    ):
        """The next image with pending items (images holding a ``fix`` first; ``queue_id`` /
        ``dataset_id`` None = any; ``only_spot``: only images with a pending spot-check item);
        every pending item of it is claimed for the reviewer. Returns (image, items) or None."""
        with self.lock:
            cutoff = time.time() - CLAIM_MINUTES * 60
            rows = self.db.execute(
                "select items.image_id, max(verdict = 'fix') fix, min(items.id) first_id,"
                " max(claimed_by is not null and claimed_by != ? and claimed_at >= ?) taken"
                " from items join queues on queues.id = items.queue_id"
                " where (? is null or items.queue_id = ?) and (? is null or queues.dataset_id = ?)"
                " and items.status = 'pending'"
                " group by items.image_id having (? = 0 or max(items.spot) = 1)"
                " order by fix desc, first_id",
                (reviewer, cutoff, queue_id, queue_id, dataset_id, dataset_id, int(only_spot)),
            ).fetchall()
            pick = None
            seen_after = after_image is None
            for r in rows:
                if r["taken"]:
                    continue
                if not seen_after:
                    seen_after = r["image_id"] == after_image
                    continue
                pick = r["image_id"]
                break
            if pick is None and after_image is not None:  # wrap around
                pick = next((r["image_id"] for r in rows if not r["taken"]), None)
            if pick is None:
                return None
            self.db.execute(
                "update items set claimed_by = ?, claimed_at = ?"
                " where (? is null or queue_id = ?) and image_id = ? and status = 'pending'",
                (reviewer, time.time(), queue_id, queue_id, pick),
            )
            self.db.commit()
        return self.image(pick), self.items_for_image(pick, queue_id)

    def items_for_image(self, image_id: str, queue_id: int | None = None) -> list[dict]:
        q = (
            "select items.*, images.path, images.width, images.height, images.source"
            " from items join images on images.id = items.image_id"
            " where image_id = ? and (? is null or queue_id = ?) order by items.id"
        )
        return [self._item(r) for r in self.db.execute(q, (image_id, queue_id, queue_id))]

    def review_summary(self) -> dict:
        """Where the data stands, across every batch (the four-step UI's header numbers)."""
        r = self.db.execute(
            "select count(distinct image_id) images, count(*) items,"
            " count(distinct case when status = 'pending' then image_id end) images_pending,"
            " count(distinct case when status = 'pending' and verdict = 'fix' then image_id end)"
            " images_fix,"
            " sum(status = 'pending') pending, sum(status = 'auto') auto,"
            " sum(status in ('approved', 'corrected')) reviewed,"
            " sum(status = 'discarded') discarded from items"
        ).fetchone()
        out = dict(r)
        out["trainable"] = (out["auto"] or 0) + (out["reviewed"] or 0)
        try:  # images that belong to a dataset count even before they have items
            out["images"] = self.db.execute(
                "select count(*) from (select image_id from dataset_images"
                " union select image_id from items)"
            ).fetchone()[0]
        except sqlite3.OperationalError:  # table not created yet (no DatasetStore)
            pass
        return out

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
        """Back to pending, as triage left it: an agreed item gets its auto label back (the
        spot-check sample carries one), any other item has none."""
        from ..autogen import auto_label

        it = self.item(item_id)
        lab = None
        if it and it.get("verdict") == "accept" and it.get("teacher") is not None:
            lab = json.dumps(auto_label(it["skill"], it["pred"], it["teacher"]), ensure_ascii=False)
        with self.lock:
            self.db.execute(
                "update items set status = 'pending', label = ?, reviewer = null,"
                " reviewed_at = null where id = ?",
                (lab, item_id),
            )
            self.db.commit()
        return self.item(item_id)


def item_row(it: dict) -> dict:
    """One reviewed item -> a training / eval row. A human's verdict is ``human``; a
    judge-accepted item nobody looked at is ``machine_reviewed`` (GOALS §4)."""
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
    lab = it["label"] or (it["pred"] if auto else None) or {}
    if it["skill"] == "detect":
        return {**base, "object": it["prompt"], "boxes": lab.get("boxes", [])}
    if it["skill"] == "point":
        return {**base, "object": it["prompt"], "points": lab.get("points", [])}
    if it["skill"] == "query":
        return {**base, "question": it["prompt"], "answer": lab.get("text", "")}
    return {**base, "length": it["prompt"], "answer": lab.get("text", "")}
