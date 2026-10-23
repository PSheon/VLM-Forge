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
from pathlib import Path

from ..data import write_rows
from .workspace import IMAGE_EXT, Workspace, item_row, now

VAL_FRACTION = 0.1

SCHEMA = """
create table if not exists datasets (
    id integer primary key,
    name text unique not null,
    created_at text not null,
    updated_at text not null,
    config text not null default '{}',
    note text not null default ''
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

    def add_bytes(self, ds_id: int, data: bytes, filename: str) -> list[str]:
        """An uploaded file: one image, or a zip of images. Returns the image ids added."""
        if filename.lower().endswith(".zip"):
            ids = []
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_EXT:
                        continue
                    im = self.ws.add_image_bytes(
                        z.read(info),
                        f"upload:{Path(filename).stem}",
                        Path(info.filename).suffix.lstrip("."),
                    )
                    self.add_image(ds_id, im["id"])
                    ids.append(im["id"])
            return ids
        ext = Path(filename).suffix.lower().lstrip(".") or "jpg"
        im = self.ws.add_image_bytes(data, "upload", ext)
        self.add_image(ds_id, im["id"])
        return [im["id"]]

    def add_folder(
        self, ds_id: int, folder: Path, limit: int | None = None, seed: int = 0
    ) -> list[str]:
        """Every image under a folder on this machine (referenced, not copied)."""
        import random

        files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXT)
        if limit and len(files) > limit:
            files = random.Random(seed).sample(files, limit)
        ids = []
        for p in files:
            im = self.ws.add_image_path(p, f"folder:{Path(folder).name}")
            self.add_image(ds_id, im["id"])
            ids.append(im["id"])
        return ids

    def image_ids(self, ds_id: int, split: str | None = None) -> list[str]:
        q = "select image_id from dataset_images where dataset_id = ?"
        args: list = [ds_id]
        if split:
            q += " and split = ?"
            args.append(split)
        return [r["image_id"] for r in self.db.execute(q + " order by added_at, image_id", args)]

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
        out["trainable_by_skill"] = {row["skill"]: row["n"] for row in ts}
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
