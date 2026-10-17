#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""输出脱敏：把核验产物里出现的已知标识符替换为研究编号。

核验报告会把**病例编号**写进 ``label``（例如 Fig. 6 的五个示例病例）。
这些编号是直接标识符，不能随仓库发布，因此在写盘前统一过一遍映射。

    from hrpca.sanitise import sanitise_text, sanitise_file
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

MAP_JSON = Path(r"C:\Users\l\Desktop\lunwen3\00_备份_ID映射_禁传\id_map_禁传.json")

#: any 7–10 digit run that is a known identifier
_PAT = re.compile(r"\b\d{7,10}\b")


def load_map() -> dict:
    try:
        # encoding="utf-8-sig" — the stored JSON carries a BOM
        d = json.loads(MAP_JSON.read_text(encoding="utf-8-sig"))
        return {str(k): str(v) for k, v in d.get("map", {}).items()}
    except Exception:
        return {}


_MAP = load_map()


def sanitise_text(text: str) -> str:
    """Replace every known identifier with its study code."""
    if not _MAP:
        return text
    return _PAT.sub(lambda m: _MAP.get(m.group(0), m.group(0)), text)


def sanitise_file(path, encoding: str = "utf-8-sig") -> int:
    """Rewrite one text/CSV/JSON file in place; return the number of replacements."""
    p = Path(path)
    if not p.is_file():
        return 0
    raw = io.open(p, encoding=encoding, errors="ignore").read()
    out = sanitise_text(raw)
    if out == raw:
        return 0
    n = sum(1 for a, b in zip(_PAT.findall(raw), _PAT.findall(out)) if a != b)
    io.open(p, "w", encoding=encoding, newline="").write(out)
    return n
