"""Your output folder is already a complete experiment record. Make it queryable.

ComfyUI embeds the whole API graph in every PNG it writes, so a directory of
renders holds every seed, every model, every LoRA and every strength that
produced them. None of it is searchable, so in practice nobody looks -- the
answer to "when did I last use this LoRA at 0.4, and what came out" is opening
files one at a time until you give up.

This walks the graphs and puts what it can resolve into SQLite.

Refuse per field, not per row
-----------------------------
An index that drops a file is useless; an index that stores a wrong value is
worse. So a render always gets a row -- path, mtime and pipeline fingerprint
are cheap and never ambiguous -- and only the genuinely unreadable fields go
``NULL``, with a status column saying which.

That is what ``lora_status`` is for. An empty LoRA list can mean two opposite
things: this render used no LoRAs, or it used LoRAs through a node shape with
no reader. Storing both as "no rows" would quietly turn the second into the
first, and any query counting LoRA usage would then be wrong in a direction
nobody would notice.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .workflow import (NONE, OK, PARTIAL, UNSUPPORTED, WorkflowError,
                       fingerprint, loras, models, prompt_text,
                       read_workflow, sampler_settings)

__all__ = ["build", "connect", "scan_one", "SCHEMA", "model_identity",
           "provenance", "compare_renders"]
__version__ = "0.1.0"

SCHEMA = """
CREATE TABLE IF NOT EXISTS render (
    id            INTEGER PRIMARY KEY,
    file          TEXT UNIQUE NOT NULL,
    path          TEXT NOT NULL,
    mtime         REAL NOT NULL,
    fingerprint   TEXT,
    checkpoint    TEXT,
    unet          TEXT,
    vae           TEXT,
    seed          INTEGER,
    steps         INTEGER,
    cfg           REAL,
    sampler       TEXT,
    scheduler     TEXT,
    denoise       REAL,
    prompt        TEXT,
    negative      TEXT,
    lora_status   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS render_lora (
    render_id       INTEGER NOT NULL REFERENCES render(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    strength_model  REAL,
    strength_clip   REAL,
    active          INTEGER NOT NULL DEFAULT 1,
    shape           TEXT NOT NULL,
    node            TEXT
);
CREATE INDEX IF NOT EXISTS render_lora_name ON render_lora(name);
CREATE INDEX IF NOT EXISTS render_fp ON render(fingerprint);
CREATE INDEX IF NOT EXISTS render_seed ON render(seed);
"""


def connect(db) -> sqlite3.Connection:
    c = sqlite3.connect(str(db))
    c.execute("PRAGMA foreign_keys = ON")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def _int(v) -> Optional[int]:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) \
        else None


def _float(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) \
        else None


def _str(v) -> Optional[str]:
    return v if isinstance(v, str) and v else None


def scan_one(png) -> Dict[str, object]:
    """Everything this can resolve from one render. Raises if unreadable."""
    png = Path(png)
    api = read_workflow(png)
    s = sampler_settings(api)
    m = models(api)
    entries, status = loras(api)
    return {
        "file": png.name,
        "path": str(png.resolve()),
        "mtime": png.stat().st_mtime,
        "fingerprint": fingerprint(api),
        "checkpoint": _str(m.get("ckpt_name")),
        "unet": _str(m.get("unet_name")),
        "vae": _str(m.get("vae_name")),
        "seed": _int(s.get("seed")),
        "steps": _int(s.get("steps")),
        "cfg": _float(s.get("cfg")),
        "sampler": _str(s.get("sampler_name")),
        "scheduler": _str(s.get("scheduler")),
        "denoise": _float(s.get("denoise")),
        "prompt": prompt_text(api, "positive"),
        "negative": prompt_text(api, "negative"),
        "lora_status": status,
        "loras": entries,
    }


def build(root, db, on_error=None, prune: bool = True) -> Tuple[int, int]:
    """Index every PNG under ``root``. Returns ``(indexed, skipped)``.

    ``prune`` drops rows whose file is no longer under ``root``. Without it a
    renamed file leaves its old row behind and every count is quietly too
    high -- an index that reports renders that do not exist is worse than one
    that is merely out of date.
    """
    root = Path(root)
    conn = connect(db)
    indexed = skipped = 0
    seen = set()
    with conn:
        for png in sorted(root.rglob("*.png")):
            try:
                row = scan_one(png)
            except (WorkflowError, OSError) as e:
                skipped += 1
                if on_error:
                    on_error(png, e)
                continue
            entries = row.pop("loras")
            # Clear the old child rows *before* the parent moves. INSERT OR
            # REPLACE deletes and re-inserts, which hands out a new rowid, so
            # deleting afterwards misses the rows still pointing at the old
            # one -- and re-indexing a folder then doubles every LoRA row.
            # ON DELETE CASCADE does not save this: SQLite ignores foreign
            # keys unless the pragma is on, which is off by default.
            old = conn.execute("SELECT id FROM render WHERE file = ?",
                               (row["file"],)).fetchone()
            if old is not None:
                conn.execute("DELETE FROM render_lora WHERE render_id = ?",
                             (old[0],))
            cols = ", ".join(row)
            conn.execute(
                "INSERT OR REPLACE INTO render (%s) VALUES (%s)"
                % (cols, ", ".join("?" * len(row))), tuple(row.values()))
            rid = conn.execute("SELECT id FROM render WHERE file = ?",
                               (row["file"],)).fetchone()[0]
            conn.execute("DELETE FROM render_lora WHERE render_id = ?", (rid,))
            for e in entries:
                conn.execute(
                    "INSERT INTO render_lora (render_id, name, "
                    "strength_model, strength_clip, active, shape, node) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (rid, e["name"], e["strength_model"], e["strength_clip"],
                     int(e["active"]), e["shape"], e.get("node")))
            seen.add(row["file"])
            indexed += 1
        if prune:
            here = str(root.resolve())
            for r in conn.execute(
                    "SELECT id, file, path FROM render").fetchall():
                if r["file"] in seen:
                    continue
                if not r["path"].startswith(here):
                    continue          # indexed from somewhere else; leave it
                conn.execute("DELETE FROM render_lora WHERE render_id = ?",
                             (r["id"],))
                conn.execute("DELETE FROM render WHERE id = ?", (r["id"],))
    conn.close()
    return indexed, skipped


def model_identity(name: str, search_paths: Iterable) -> Optional[dict]:
    """Size and mtime for a model file, or ``None`` if it is not found.

    A filename is not an identity. Two people's ``anime_v3.safetensors`` are
    different files, and yours changes when you re-download it. Size and mtime
    are weak, but they are checkable, and ``None`` is an honest answer where a
    guess is not.
    """
    for base in search_paths:
        for p in Path(base).rglob(Path(name).name):
            try:
                st = p.stat()
            except OSError:
                continue
            return {"path": str(p), "size": st.st_size, "mtime": st.st_mtime}
    return None


def provenance(png, search_paths: Iterable = ()) -> Dict[str, object]:
    """How one render came to be, read from the render itself.

    Never from a side-car or a notebook. A second copy of the truth drifts
    from the first, and the copy inside the file is the one that was submitted
    to ComfyUI.
    """
    row = scan_one(png)
    ids = {}
    for key in ("checkpoint", "unet", "vae"):
        if row.get(key) and search_paths:
            ids[key] = model_identity(row[key], search_paths)
    row["model_identity"] = ids
    return row


#: Fields whose difference makes a two-render comparison uncontrolled.
_CONTROLLED = ("checkpoint", "unet", "vae", "seed", "steps", "cfg",
               "sampler", "scheduler", "denoise", "negative")


def compare_renders(a, b) -> Dict[str, object]:
    """What differs between two renders, split into settings and LoRAs.

    The useful question is rarely "what are these two" -- it is "what did I
    change". Reporting whether exactly one thing moved is the part that makes
    a later conclusion about *why* the images differ either supportable or
    not.
    """
    ra, rb = scan_one(a), scan_one(b)
    settings = {k: (ra.get(k), rb.get(k)) for k in _CONTROLLED
                if ra.get(k) != rb.get(k)}

    def key(entries):
        return {(e["name"], e["strength_model"]) for e in entries
                if e["active"]}

    la, lb = key(ra["loras"]), key(rb["loras"])
    lora_changes = sorted(
        ("removed", n, s) for n, s in la - lb) + sorted(
        ("added", n, s) for n, s in lb - la)

    prompt_changed = (ra.get("prompt") or "") != (rb.get("prompt") or "")
    n = len(settings) + len(lora_changes) + (1 if prompt_changed else 0)
    unreadable = [r["lora_status"] for r in (ra, rb)
                  if r["lora_status"] in (UNSUPPORTED, PARTIAL)]
    return {
        "settings": settings,
        "loras": lora_changes,
        "prompt_changed": prompt_changed,
        "changed": n,
        "single_variable": n == 1 and not unreadable,
        "lora_status": (ra["lora_status"], rb["lora_status"]),
        "caveat": ("at least one render's LoRA stack could not be read, so "
                   "this comparison may be missing changes"
                   if unreadable else ""),
    }
