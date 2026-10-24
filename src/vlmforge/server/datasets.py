"""Datasets: the first-class thing a person builds (GOALS §4).

A dataset has a name, images from any mix of sources (uploads, zips, folders on this
machine), an AI-teacher configuration (automatic questions and/or fixed prompts) and, once
reviewed, the confirmed items. Every image gets a split on entry: 10 % ``val`` (by a hash of
the image id, so it never changes), the rest ``train``. Training reads the train split;
evaluation uses the val split as the dataset's own test.

Tables (in the workspace SQLite, next to ``images`` / ``queues`` / ``items``)::

    datasets        id, name, created_at, updated_at, config (json), note
    dataset_images  dataset_id, image_id, split ('train' | 'val'), added_at

``config``: ``{"mode": "auto" | "fixed" | "both", "fixed": "<lines>", "topics": [...],
"focus": "...", "skills": [...]}``; ``fixed`` lines look like ``detect: person, helmet`` /
``query: Is anyone without a helmet?`` / ``point: person`` / ``caption: normal``.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from ..data import write_rows
from .workspace import IMAGE_EXT, Workspace, item_row, now, sha256_file, signature

VAL_FRACTION = 0.1
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts", ".webm", ".gif"}
VIDEO_EVERY_S = 5.0
TRASH_DAYS = 7  # removed images can be restored this long; then their files go
# Near-duplicate: the 99th-percentile pixel difference of two 64x36 RGB signatures.
# Calibrated on the factory CCTV frames (2026-10-24): <= 18 was the same empty scene hours
# apart, 20 a moved box, 29 a ladder that appeared. 12 only drops frames where nothing moved.
DUP_P99 = 12

SCHEMA = """
create table if not exists datasets (
    id integer primary key,
    name text unique not null,
    created_at text not null,
    updated_at text not null,
    config text not null default '{}',
    note text not null default ''
);
create table if not exists trash (    -- removed images, restorable for TRASH_DAYS
    id integer primary key,
    dataset_id integer not null,
    created_at text not null,
    payload text not null           -- json: dataset_images rows, items rows, batch image ids
);
create table if not exists dataset_images (
    dataset_id integer not null references datasets(id),
    image_id text not null references images(id),
    split text not null default 'train',
    added_at text not null,
    primary key (dataset_id, image_id)
);
"""


def split_for(image_id: str) -> str:
    """Deterministic 10 % validation split from the image's own id (its sha256)."""
    return "val" if int(image_id[:8], 16) % 1000 < VAL_FRACTION * 1000 else "train"


class DatasetStore:
    def __init__(self, ws: Workspace):
        self.ws = ws
        ws.db.executescript(SCHEMA)
        have = {r["name"] for r in ws.db.execute("pragma table_info(queues)")}
        if "dataset_id" not in have:
            ws.db.execute("alter table queues add column dataset_id integer")
        ws.db.commit()

    @property
    def db(self):
        return self.ws.db

    # ------------------------------------------------------------------ crud
    def create(self, name: str, config: dict | None = None, note: str = "") -> dict:
        name = name.strip()
        if not name:
            raise ValueError("名稱不能空白")
        with self.ws.lock:
            self.db.execute(
                "insert into datasets (name, created_at, updated_at, config, note)"
                " values (?, ?, ?, ?, ?)",
                (name, now(), now(), json.dumps(config or {}, ensure_ascii=False), note),
            )
            self.db.commit()
        return self.get_by_name(name)

    def update(self, ds_id: int, config: dict | None = None, name: str | None = None) -> dict:
        with self.ws.lock:
            if config is not None:
                self.db.execute(
                    "update datasets set config = ?, updated_at = ? where id = ?",
                    (json.dumps(config, ensure_ascii=False), now(), ds_id),
                )
            if name:
                self.db.execute(
                    "update datasets set name = ?, updated_at = ? where id = ?",
                    (name.strip(), now(), ds_id),
                )
            self.db.commit()
        return self.get(ds_id)

    def delete(self, ds_id: int) -> None:
        """Removes the dataset and its membership rows; images and items stay in the workspace."""
        with self.ws.lock:
            self.db.execute("delete from dataset_images where dataset_id = ?", (ds_id,))
            self.db.execute("update queues set dataset_id = null where dataset_id = ?", (ds_id,))
            self.db.execute("delete from datasets where id = ?", (ds_id,))
            self.db.commit()

    def _row(self, row) -> dict:
        if row is None:
            raise KeyError("no such dataset")
        d = dict(row)
        d["config"] = json.loads(d["config"] or "{}")
        d.update(self.counts(d["id"]))
        d["thumbs"] = [  # the list's thumbnail strip: the first few images added
            r[0]
            for r in self.db.execute(
                "select image_id from dataset_images where dataset_id = ?"
                " order by added_at, image_id limit 6",
                (d["id"],),
            )
        ]
        return d

    def get(self, ds_id: int) -> dict:
        return self._row(
            self.db.execute("select * from datasets where id = ?", (ds_id,)).fetchone()
        )

    def get_by_name(self, name: str) -> dict:
        return self._row(
            self.db.execute("select * from datasets where name = ?", (name,)).fetchone()
        )

    def list(self) -> list[dict]:
        return [
            self._row(r) for r in self.db.execute("select * from datasets order by updated_at desc")
        ]

    # ------------------------------------------------------------------ images
    def add_image(self, ds_id: int, image_id: str) -> str:
        """Adds an existing workspace image; returns its split."""
        self.get(ds_id)  # KeyError for a dataset that does not exist (SQLite has no FK on)
        split = split_for(image_id)
        with self.ws.lock:
            self.db.execute(
                "insert or ignore into dataset_images values (?, ?, ?, ?)",
                (ds_id, image_id, split, now()),
            )
            self.db.execute("update datasets set updated_at = ? where id = ?", (now(), ds_id))
            self.db.commit()
        return split

    # Adding images. Every path goes through ``_Adder``: an image that is (nearly) the same
    # frame as one already in the dataset is skipped before it is stored (GOALS §4).
    def add_bytes(self, ds_id: int, data: bytes, filename: str) -> dict:
        """An uploaded file: one image, or a zip of images (an animated GIF is a video)."""
        add = _Adder(self, ds_id, f"upload:{Path(filename).stem}")
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    ext = Path(info.filename).suffix.lower()
                    if not info.is_dir() and ext in IMAGE_EXT:
                        add.image_bytes(z.read(info), ext.lstrip("."))
            return add.result()
        ext = Path(filename).suffix.lower().lstrip(".") or "jpg"
        add.source = "upload"
        add.image_bytes(data, ext)
        return add.result()

    def add_video(self, ds_id: int, path: Path, every_s: float, name: str | None = None) -> dict:
        """One frame every ``every_s`` seconds of a video (or an animated GIF), stored in the
        workspace like an upload."""
        add = _Adder(self, ds_id, f"video:{name or Path(path).stem}")
        for im in video_frames(Path(path), every_s):
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=92)
            add.image_bytes(buf.getvalue(), "jpg", im)
        return add.result()

    def add_folder(
        self,
        ds_id: int,
        folder: Path,
        limit: int | None = None,
        seed: int = 0,
        every_s: float = VIDEO_EVERY_S,
    ) -> dict:
        """Every image under a folder on this machine (referenced, not copied) and frames
        from every video in it (stored in the workspace)."""
        import random

        files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXT)
        videos = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXT)
        if limit and len(files) > limit:
            files = random.Random(seed).sample(files, limit)
        add = _Adder(self, ds_id, f"folder:{Path(folder).name}")
        for p in files:
            add.image_path(p)
        out = add.result()
        for v in videos:
            r = self.add_video(ds_id, v, every_s)
            out["image_ids"] += r["image_ids"]
            out["skipped"] += r["skipped"]
        out["added"] = len(out["image_ids"])
        return out

    def duplicates(self, ds_id: int) -> list[str]:
        """Images that repeat an earlier image of the dataset (same frame again): the later
        copy of each, ready to be picked and removed."""
        ids = self.image_ids(ds_id)  # in the order they were added
        sigs = self.ws.signatures(ids)
        seen = _SigIndex([])
        dup = []
        for i in ids:
            if i not in sigs:
                continue
            if seen.repeats(sigs[i]):
                dup.append(i)
            else:
                seen.add(sigs[i])
        return dup

    def image_ids(self, ds_id: int, split: str | None = None) -> list[str]:
        q = "select image_id from dataset_images where dataset_id = ?"
        args: list = [ds_id]
        if split:
            q += " and split = ?"
            args.append(split)
        return [r["image_id"] for r in self.db.execute(q + " order by added_at, rowid", args)]

    def remove_images(self, ds_id: int, image_ids: list[str]) -> dict:
        """Takes images out of the dataset together with their questions in it (reviewed or
        not), and drops them from its batches' configs so 「接著跑」 does not ask them again.
        What was removed goes to ``trash`` for ``TRASH_DAYS`` (``restore`` puts it back);
        ``purge`` then deletes the files this workspace stored once nothing uses them. A file
        in a GB300 folder is never touched."""
        self.get(ds_id)
        ids = sorted(set(image_ids))
        if not ids:
            return {"removed": 0, "items": 0, "trash_id": None}
        marks = ",".join("?" * len(ids))
        with self.ws.lock:
            qids = [
                r[0]
                for r in self.db.execute("select id from queues where dataset_id = ?", (ds_id,))
            ]
            members = [
                dict(r)
                for r in self.db.execute(
                    f"select * from dataset_images where dataset_id = ? and image_id in ({marks})",
                    (ds_id, *ids),
                )
            ]
            items, batches = [], {}
            if qids:
                qm = ",".join("?" * len(qids))
                where = f"image_id in ({marks}) and queue_id in ({qm})"
                items = [
                    dict(r)
                    for r in self.db.execute(f"select * from items where {where}", (*ids, *qids))
                ]
                self.db.execute(f"delete from items where {where}", (*ids, *qids))
                for qid, cfg in self.db.execute(
                    f"select id, config from queues where id in ({qm}) and config is not null", qids
                ).fetchall():
                    gone = [i for i in json.loads(cfg).get("image_ids", []) if i in set(ids)]
                    if gone:
                        batches[str(qid)] = gone
                self._drop_from_batches(qids, ids)
            self.db.execute(
                f"delete from dataset_images where dataset_id = ? and image_id in ({marks})",
                (ds_id, *ids),
            )
            payload = {"images": members, "items": items, "batches": batches}
            trash_id = self.db.execute(
                "insert into trash (dataset_id, created_at, payload) values (?, ?, ?)",
                (ds_id, now(), json.dumps(payload, ensure_ascii=False)),
            ).lastrowid
            self.db.execute("update datasets set updated_at = ? where id = ?", (now(), ds_id))
            self.db.commit()
        return {"removed": len(members), "items": len(items), "trash_id": trash_id}

    def restore(self, ds_id: int, trash_id: int) -> dict:
        """Undo one ``remove_images``: the images, their questions (same ids, labels and
        reviews) and their place in the batches come back."""
        with self.ws.lock:
            row = self.db.execute(
                "select payload from trash where id = ? and dataset_id = ?", (trash_id, ds_id)
            ).fetchone()
            if row is None:
                raise KeyError("nothing to restore")
            p = json.loads(row[0])
            for table, rows in (("dataset_images", p["images"]), ("items", p["items"])):
                for r in rows:
                    cols = ",".join(r)
                    self.db.execute(
                        f"insert or ignore into {table} ({cols}) values ({','.join('?' * len(r))})",
                        list(r.values()),
                    )
            for qid, back in p["batches"].items():
                cfg = self.db.execute(
                    "select config from queues where id = ?", (int(qid),)
                ).fetchone()
                if cfg and cfg[0]:
                    c = json.loads(cfg[0])
                    c["image_ids"] = c.get("image_ids", []) + [
                        i for i in back if i not in c.get("image_ids", [])
                    ]
                    self.db.execute(
                        "update queues set config = ? where id = ?",
                        (json.dumps(c, ensure_ascii=False), int(qid)),
                    )
            self.db.execute("delete from trash where id = ?", (trash_id,))
            self.db.execute("update datasets set updated_at = ? where id = ?", (now(), ds_id))
            self.db.commit()
        return {"restored": len(p["images"]), "items": len(p["items"])}

    def purge(self, days: float = TRASH_DAYS) -> int:
        """Forget removals older than ``days`` and delete the files this workspace stored for
        them once nothing references them (uploads / video frames; never a GB300 folder)."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        files = 0
        with self.ws.lock:
            old = self.db.execute("select id, payload from trash where created_at <= ?", (cutoff,))
            old = old.fetchall()
            ids = {r["image_id"] for t in old for r in json.loads(t[1])["images"]}
            root = self.ws.images_dir.resolve()
            for i in sorted(ids):
                busy = self.db.execute(
                    "select (select count(*) from dataset_images where image_id = ?)"
                    " + (select count(*) from items where image_id = ?)",
                    (i, i),
                ).fetchone()[0]
                row = self.db.execute("select path from images where id = ?", (i,)).fetchone()
                if busy or row is None or Path(row[0]).resolve().parent != root:
                    continue
                Path(row[0]).unlink(missing_ok=True)
                for c in (self.ws.root / "cache").glob(f"{i}-w*.jpg"):
                    c.unlink(missing_ok=True)
                self.db.execute("delete from images where id = ?", (i,))
                self.db.execute("delete from image_sigs where image_id = ?", (i,))
                files += 1
            self.db.executemany("delete from trash where id = ?", [(t[0],) for t in old])
            self.db.commit()
        return files

    def _drop_from_batches(self, qids: list[int], image_ids: list[str]) -> None:
        """Forget images in these batches' configs, so an old batch neither counts them as
        unfinished nor asks them again on 「接著跑」 (caller holds the lock)."""
        drop = set(image_ids)
        qm = ",".join("?" * len(qids))
        for qid, cfg in self.db.execute(
            f"select id, config from queues where id in ({qm}) and config is not null", qids
        ).fetchall():
            c = json.loads(cfg)
            if drop & set(c.get("image_ids", [])):
                c["image_ids"] = [i for i in c["image_ids"] if i not in drop]
                self.db.execute(
                    "update queues set config = ? where id = ?",
                    (json.dumps(c, ensure_ascii=False), qid),
                )

    def reset_for_relabel(self, ds_id: int, image_ids: list[str]) -> dict:
        """Before asking these images again with the current teacher settings: their machine
        questions (自動通過 / 待看) in this dataset go; what a human saved, discarded or added by
        hand stays and is not asked twice (see ``asked``)."""
        self.get(ds_id)
        ids = sorted(set(image_ids))
        with self.ws.lock:
            qids = [
                r[0]
                for r in self.db.execute("select id from queues where dataset_id = ?", (ds_id,))
            ]
            if not ids or not qids:
                return {"dropped": 0, "kept": 0}
            marks, qm = ",".join("?" * len(ids)), ",".join("?" * len(qids))
            where = f"image_id in ({marks}) and queue_id in ({qm})"
            dropped = self.db.execute(
                f"delete from items where {where} and status in ('pending', 'auto')"
                " and origin != 'workbench'",
                (*ids, *qids),
            ).rowcount
            kept = self.db.execute(f"select count(*) from items where {where}", (*ids, *qids))
            self._drop_from_batches(qids, ids)
            self.db.commit()
        return {"dropped": dropped, "kept": kept.fetchone()[0]}

    def asked(self, ds_id: int, image_id: str) -> set[tuple[str, str]]:
        """(skill, prompt) pairs this image already has in the dataset."""
        rows = self.db.execute(
            "select items.skill, items.prompt from items join queues on queues.id = items.queue_id"
            " where queues.dataset_id = ? and items.image_id = ?",
            (ds_id, image_id),
        ).fetchall()
        return {(r[0], r[1]) for r in rows}

    def unlabelled_image_ids(self, ds_id: int) -> list[str]:
        """Images of the dataset that no batch has produced items for yet."""
        rows = self.db.execute(
            "select di.image_id from dataset_images di"
            " left join (select distinct items.image_id from items"
            "            join queues on queues.id = items.queue_id"
            "            where queues.dataset_id = ?) lab on lab.image_id = di.image_id"
            " where di.dataset_id = ? and lab.image_id is null order by di.added_at",
            (ds_id, ds_id),
        ).fetchall()
        return [r["image_id"] for r in rows]

    # ------------------------------------------------------------------ counts / rows
    def counts(self, ds_id: int) -> dict:
        r = self.db.execute(
            "select count(*) images, sum(split = 'val') val from dataset_images"
            " where dataset_id = ?",
            (ds_id,),
        ).fetchone()
        c = self.db.execute(
            "select count(*) items, sum(items.status = 'pending') pending,"
            " sum(items.status = 'auto') auto,"
            " sum(items.status in ('approved', 'corrected')) reviewed,"
            " sum(items.status = 'discarded') discarded,"
            " count(distinct case when items.status = 'pending' then items.image_id end)"
            " images_pending,"
            " sum(items.skill = 'detect') detect, sum(items.skill = 'point') point,"
            " sum(items.skill = 'query') query, sum(items.skill = 'caption') caption"
            " from items join queues on queues.id = items.queue_id where queues.dataset_id = ?",
            (ds_id,),
        ).fetchone()
        ti = self.db.execute(
            "select count(distinct items.image_id) from items"
            " join queues on queues.id = items.queue_id"
            " join dataset_images di on di.image_id = items.image_id"
            " and di.dataset_id = queues.dataset_id"
            " where queues.dataset_id = ? and di.split = 'train'"
            " and items.status in ('auto', 'approved', 'corrected')",
            (ds_id,),
        ).fetchone()[0]
        ts = self.db.execute(
            "select items.skill, count(*) n from items"
            " join queues on queues.id = items.queue_id"
            " join dataset_images di on di.image_id = items.image_id"
            " and di.dataset_id = queues.dataset_id"
            " where queues.dataset_id = ? and di.split = 'train'"
            " and items.status in ('auto', 'approved', 'corrected') group by items.skill",
            (ds_id,),
        ).fetchall()
        out = {"images": r["images"] or 0, "val_images": r["val"] or 0}
        out.update({k: (c[k] or 0) for k in c.keys()})
        out["trainable"] = out["auto"] + out["reviewed"]
        out["trainable_images"] = ti or 0  # train-split images with ≥1 confirmed item
        out["val_rows"] = self.db.execute(  # confirmed questions in the val split (the val curve)
            "select count(*) from items join queues on queues.id = items.queue_id"
            " join dataset_images di on di.image_id = items.image_id"
            " and di.dataset_id = queues.dataset_id"
            " where queues.dataset_id = ? and di.split = 'val'"
            " and items.status in ('auto', 'approved', 'corrected')",
            (ds_id,),
        ).fetchone()[0]
        out["trainable_by_skill"] = {row["skill"]: row["n"] for row in ts}
        # the spot check measures the teacher: of the agreed questions a human re-checked, how
        # many did they leave as they were (approved) vs change or discard
        sp = self.db.execute(
            "select sum(items.spot = 1) total,"
            " sum(items.spot = 1 and items.status = 'pending') pending,"
            " sum(items.spot = 1 and items.status = 'approved') agree,"
            " sum(items.spot = 1 and items.status in ('corrected', 'discarded')) changed"
            " from items join queues on queues.id = items.queue_id where queues.dataset_id = ?",
            (ds_id,),
        ).fetchone()
        out["spot"] = {k: sp[k] or 0 for k in ("total", "pending", "agree", "changed")}
        return out

    def rows(self, ds_id: int, split: str) -> list[dict]:
        """Confirmed rows (human + auto) of one split, in the training / eval row format."""
        wanted = set(self.image_ids(ds_id, split))
        out = []
        for it in self._confirmed_items(ds_id):
            if it["image_id"] in wanted:
                out.append(item_row(it))
        return out

    def _confirmed_items(self, ds_id: int) -> list[dict]:
        q = (
            "select items.*, images.path, images.width, images.height, images.source"
            " from items join images on images.id = items.image_id"
            " join queues on queues.id = items.queue_id"
            " where queues.dataset_id = ? and items.status in ('approved', 'corrected', 'auto')"
            " order by items.id"
        )
        return [self.ws._item(r) for r in self.db.execute(q, (ds_id,))]

    def attach_queue(self, queue_id: int, ds_id: int) -> None:
        with self.ws.lock:
            self.db.execute("update queues set dataset_id = ? where id = ?", (ds_id, queue_id))
            self.db.commit()

    def export(self, ds_id: int, out_dir: Path, with_images: bool = True) -> dict:
        """``<name>.jsonl`` (train + val rows, each with ``split``) and, optionally, a zip of
        the images named by their id so the jsonl stays valid on another machine."""
        ds = self.get(ds_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for split in ("train", "val"):
            for r in self.rows(ds_id, split):
                rows.append({**r, "split": split})
        jsonl = out_dir / f"{ds['name']}.jsonl"
        write_rows(jsonl, rows, ensure_ascii=False)
        res = {"jsonl": str(jsonl), "rows": len(rows)}
        if with_images:
            zpath = out_dir / f"{ds['name']}-images.zip"
            seen: set[str] = set()
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
                for r in rows:
                    p = Path(r["image"])
                    if str(p) in seen or not p.exists():
                        continue
                    seen.add(str(p))
                    z.write(
                        p,
                        f"images/{hashlib.sha256(str(p).encode()).hexdigest()[:16]}{p.suffix.lower()}",
                    )
            res["zip"] = str(zpath)
            res["images"] = len(seen)
        return res


class _SigIndex:
    """Signatures seen so far; ``repeats(sig)`` = some earlier image is the same frame."""

    def __init__(self, sigs: list[bytes]):
        flat = np.frombuffer(b"".join(sigs), np.uint8)
        self.m = flat.reshape(len(sigs), 64 * 36 * 3).astype(np.int16)

    def repeats(self, sig: bytes) -> bool:
        if not len(self.m):
            return False
        d = np.abs(self.m - np.frombuffer(sig, np.uint8).astype(np.int16))
        near = d.mean(axis=1) <= DUP_P99  # p99 >= mean, so only these can be duplicates
        return bool(near.any() and (np.percentile(d[near], 99, axis=1) <= DUP_P99).any())

    def add(self, sig: bytes) -> None:
        self.m = np.concatenate([self.m, np.frombuffer(sig, np.uint8).astype(np.int16)[None]])


class _Adder:
    """Adds images to one dataset, skipping ones it already has (same file, or nearly the
    same frame) before anything is written."""

    def __init__(self, store: DatasetStore, ds_id: int, source: str):
        store.get(ds_id)  # KeyError for a dataset that does not exist
        self.store, self.ds_id, self.source = store, ds_id, source
        have = store.image_ids(ds_id)
        self.have = set(have)
        self.index = _SigIndex(list(store.ws.signatures(have).values()))
        self.ids: list[str] = []
        self.skipped = 0

    def _fresh(self, image_id: str, im: Image.Image) -> bytes | None:
        if image_id in self.have:
            self.skipped += 1
            return None
        sig = signature(im)
        if self.index.repeats(sig):
            self.skipped += 1
            return None
        return sig

    def image_bytes(self, data: bytes, ext: str, im: Image.Image | None = None) -> None:
        sig = self._fresh(hashlib.sha256(data).hexdigest(), im or Image.open(io.BytesIO(data)))
        if sig is not None:
            self._keep(self.store.ws.add_image_bytes(data, self.source, ext)["id"], sig)

    def image_path(self, path: Path) -> None:
        image_id = sha256_file(path)
        sig = self._fresh(image_id, Image.open(path))
        if sig is not None:
            self._keep(self.store.ws.add_image_path(path, self.source, image_id)["id"], sig)

    def _keep(self, image_id: str, sig: bytes) -> None:
        self.store.ws.put_signature(image_id, sig)
        self.store.add_image(self.ds_id, image_id)
        self.index.add(sig)
        self.have.add(image_id)
        self.ids.append(image_id)

    def result(self) -> dict:
        return {"added": len(self.ids), "skipped": self.skipped, "image_ids": self.ids}


def video_frames(path: Path, every_s: float):
    """One PIL frame every ``every_s`` seconds. GIFs go through PIL (their frame durations);
    videos through PyAV, seeking to each target time instead of decoding everything."""
    every_s = max(0.2, float(every_s))
    if path.suffix.lower() == ".gif":
        im, t, nxt = Image.open(path), 0.0, 0.0
        for k in range(getattr(im, "n_frames", 1)):
            im.seek(k)
            if t + 1e-6 >= nxt:
                yield im.convert("RGB")
                nxt = t + every_s
            t += (im.info.get("duration") or 100) / 1000
        return
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        dur = float(c.duration / 1e6) if c.duration else float((s.duration or 0) * s.time_base)
        t = 0.0
        while t <= dur + 1e-6:
            c.seek(int(t / s.time_base) + (s.start_time or 0), stream=s, backward=True)
            for frame in c.decode(s):
                ft = frame.time if frame.time is not None else 0.0
                if ft + 1e-3 >= t + (float(s.start_time or 0) * s.time_base):
                    yield frame.to_image()
                    break
            else:
                break
            t += every_s
