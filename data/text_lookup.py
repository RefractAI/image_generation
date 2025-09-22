"""Utilities for retrieving cached text metadata.

Provides concrete implementations for two sources with minimal error handling:
- ShelveTextLookup: reads captions from two Python shelve DB files
  (tags/labels and natural language prompts).
- TagsDBTextLookup: reads tags from a SQLite database table.

Also exposes a factory ``create_text_lookup(cfg)`` that builds from
``CacheTextLookupConfig``.
"""

from __future__ import annotations

from __future__ import annotations

import shelve
import sqlite3
from typing import Optional
import json
from abc import ABC, abstractmethod


class TextLookupBase(ABC):

    @abstractmethod
    def lookup(self, key: object) -> tuple[str, str]:  # pragma: no cover - interface definition
        raise NotImplementedError
    
class ShelveTextLookup(TextLookupBase):

    def __init__(self, nlprompt_path: str, tags_path: Optional[str] = None) -> None:
        self._nl_path = str(nlprompt_path)
        self._tags_path = str(tags_path) if tags_path else f"{nlprompt_path}_tags"
        self._nl_shelf = shelve.open(self._nl_path, flag="r")
        self._tags_shelf = shelve.open(self._tags_path, flag="r")

    def lookup(self, key: object) -> tuple[str, str]:
        s_key = str(key)
        value = self._tags_shelf.get(s_key, "")
        label = str(value) if value is not None else ""
        alt_value = self._nl_shelf.get(s_key, "")
        alt = str(alt_value) if alt_value is not None else ""
        return label, alt

class TagsDBTextLookup(TextLookupBase):

    def __init__(
        self,
        db_path: str,
        table: str = "tags_lookup",
        key_column: str = "post_id",
        tags_column: str = "tags",
    ) -> None:
        self._db_path = str(db_path)
        self._table = str(table)
        self._key_col = str(key_column)
        self._tags_col = str(tags_column)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)

    def lookup(self, key: object) -> tuple[str, str]:
        key_int = int(key)
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT {self._tags_col} FROM {self._table} WHERE {self._key_col} = ?",
            (key_int,),
        )
        row = cur.fetchone()
        return (row[0] if row and row[0] is not None else "", "")


class JsonlTextLookup(TextLookupBase):
    """Lookup captions from a JSONL file with objects like:
    {"id": 2617124, "caption_cogvlm": "..."}

    Returns ("", caption) so the caption becomes the alt_caption.
    """

    def __init__(self, path: str, id_field: str = "id", caption_field: str = "caption_cogvlm") -> None:
        self._map: dict[str, str] = {}
        self._id_field = id_field
        self._caption_field = caption_field
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = str(obj[self._id_field])
                caption = str(obj.get(self._caption_field, ""))
                self._map[key] = caption

    def lookup(self, key: object) -> tuple[str, str]:
        s_key = str(key)
        return "", self._map.get(s_key, "")

def create_text_lookup(cfg) -> TextLookupBase:
    if cfg is None or cfg.type is None or cfg.path is None:
        raise ValueError("Text lookup requires both 'type' and 'path' in config")
    t = str(cfg.type).lower()
    p = str(cfg.path)
    if t == "shelve":
        if "|" in p:
            nl, tags = p.split("|", 1)
            return ShelveTextLookup(nl.strip(), tags.strip())
        return ShelveTextLookup(p)
    if t in ("sqlite", "tagsdb"):
        return TagsDBTextLookup(p)
    if t == "jsonl":
        return JsonlTextLookup(p)
    raise ValueError(f"Unsupported text lookup type: {cfg.type}")


__all__ = [
    "TextLookupBase",
    "ShelveTextLookup",
    "TagsDBTextLookup",
    "JsonlTextLookup",
    "create_text_lookup",
]
