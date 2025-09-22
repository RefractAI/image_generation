import json
import os
import shelve
import sqlite3
from pathlib import Path

import pytest

from data.text_lookup import ShelveTextLookup, TagsDBTextLookup, JsonlTextLookup, create_text_lookup
from config import CacheTextLookupConfig


def test_shelve_text_lookup_reads_label_and_alt(tmp_path: Path):
    nl_path = str(tmp_path / "nl_shelf")
    tags_path = str(tmp_path / "tags_shelf")

    with shelve.open(nl_path, flag="n") as nl:
        nl["k1"] = "a witty alt"
    with shelve.open(tags_path, flag="n") as tg:
        tg["k1"] = "class label"

    lookup = ShelveTextLookup(nl_path, tags_path)
    label, alt = lookup.lookup("k1")
    assert label == "class label"
    assert alt == "a witty alt"

    # Factory pathway
    cfg = CacheTextLookupConfig(type="shelve", path=f"{nl_path}|{tags_path}")
    f_lookup = create_text_lookup(cfg)
    f_label, f_alt = f_lookup.lookup("k1")
    assert (f_label, f_alt) == (label, alt)


def test_tagsdb_text_lookup_reads_label(tmp_path: Path):
    db_path = tmp_path / "tags_lookup.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE tags_lookup (post_id INTEGER PRIMARY KEY, tags TEXT)")
    cur.execute("INSERT INTO tags_lookup (post_id, tags) VALUES (?, ?)", (123, "flower"))
    conn.commit()
    conn.close()

    lookup = TagsDBTextLookup(str(db_path))
    label, alt = lookup.lookup(123)
    assert label == "flower"
    assert alt == ""

    cfg = CacheTextLookupConfig(type="sqlite", path=str(db_path))
    f_lookup = create_text_lookup(cfg)
    f_label, f_alt = f_lookup.lookup(123)
    assert (f_label, f_alt) == (label, alt)


def test_jsonl_text_lookup_reads_alt(tmp_path: Path):
    jsonl_path = tmp_path / "caps.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": 42, "caption_cogvlm": "hello world"}) + "\n")

    lookup = JsonlTextLookup(str(jsonl_path))
    label, alt = lookup.lookup(42)
    assert label == ""
    assert alt == "hello world"

