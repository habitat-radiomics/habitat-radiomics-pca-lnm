#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究专用编号（pseudonymisation of the admission-number key).

Background
----------
The hospital identifier that links the imaging archive to ``Clinic.csv`` and
``label.csv`` is an **admission number** and is therefore a *direct identifier*
under the IRB / journal data-sharing rules.  It must never appear in a public
repository (see ``verify/scan_identifiers.py``).

This module replaces every such identifier with a study code ``S0001`` …
``S1125``.  The replacement is derived **deterministically** from the source
tables — sort the identifiers numerically and number them in order — so the same
inputs always give the same codes and a reviewer can rebuild the table.

The mapping itself is written **outside the repository**, to the directory
named by ``HRPCA_ID_MAP_DIR``, and must never be committed.

Usage
-----
    from hrpca.pseudonym import load_map, apply_to_frame, apply_to_file

``load_map()`` rebuilds the mapping from the authoritative sources.  If the
out-of-repo mapping file exists it is used as a consistency check.
"""
from __future__ import annotations

import os

import csv
import io
import json
import re
from pathlib import Path

#: identifier columns, by exact name
ID_COLUMNS = ("ID", "real_id", "id", "patient_id", "PID", "row_index")

#: a cell that *is* a hospital identifier
ID_CELL = re.compile(r"^\d{7,10}$")

#: the same, but carrying the imaging file extension (feature tables store
#: ``<admission-number>.nii.gz`` rather than a bare number)
ID_CELL_SUFFIXED = re.compile(r"^(\d{7,10})(\.nii\.gz)$", re.I)

#: where the (never-committed) mapping lives
MAP_DIR = Path(os.environ.get("HRPCA_ID_MAP_DIR", ""))
MAP_JSON = Path(os.environ.get("HRPCA_ID_MAP", ""))
MAP_CSV = MAP_DIR / "ID_MAP.csv"

#: authoritative sources of the identifier universe (repo-relative handled by caller)
DEFAULT_SOURCES: list = []


def _bare(v: str) -> str:
    """Strip an imaging file extension so identifiers share one key space.

    ``label.csv`` stores ``<admission-number>.nii.gz`` while ``Clinic.csv``
    stores the bare number; both denote the same patient.
    """
    m = ID_CELL_SUFFIXED.match(v.strip())
    return m.group(1) if m else v.strip()


def _pairs(sources) -> list:
    out = []
    for path, col in sources:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"identifier source missing: {p}")
        with io.open(p, encoding="utf-8-sig", errors="ignore") as fh:
            rd = csv.DictReader(fh)
            if col not in (rd.fieldnames or []):
                raise ValueError(f"{p} has no column {col!r} (has {rd.fieldnames})")
            for r in rd:
                v = _bare(r.get(col) or "")
                if v:
                    out.append(v)
    return out


def build_map(sources) -> dict:
    """Deterministic ``{admission_number: 'S0001', …}``.

    Identifiers are sorted **numerically** (all are digit strings after
    :func:`_bare`), so the table is reproducible from the source files alone.
    """
    ids = _pairs(sources)
    bad = [v for v in ids if not ID_CELL.match(v)]
    if bad:
        raise ValueError(f"non-identifier values in the ID sources, e.g. {bad[:3]}")
    uniq = sorted(set(ids), key=int)
    if len(uniq) > 9999:
        raise ValueError(f"{len(uniq)} identifiers exceeds the S0001–S9999 space")
    return {old: f"S{i:04d}" for i, old in enumerate(uniq, start=1)}


def load_map(sources):
    """Return the identifier mapping.

    The stored table in :data:`MAP_DIR` is authoritative when present, because
    the pipeline is re-runnable: after a first pass the repository's
    ``labels.csv`` already holds study codes, so the mapping can no longer be
    rebuilt from it.  When the stored table is absent the mapping is rebuilt
    from the authoritative sources and cross-checked.
    """
    if MAP_JSON.exists():
        try:
            stored = json.loads(MAP_JSON.read_text(encoding="utf-8")).get("map", {})
            if stored:
                return stored
        except Exception:
            pass
    m = build_map(sources)
    return m


def load_map_strict(sources):
    """Build from the sources and require agreement with the stored copy."""
    m = build_map(sources)
    if MAP_JSON.exists():
        stored = json.loads(MAP_JSON.read_text(encoding="utf-8")).get("map", {})
        if stored and stored != m:
            raise ValueError(
                "identifier mapping differs from the stored out-of-repo copy "
                f"({MAP_JSON}); refusing to pseudonymise with an unexpected map"
            )
    return m


def map_value(v, mapping: dict):
    """Replace one identifier value, keeping any ``.nii.gz`` suffix.

    The mapping keys are **bare** admission numbers, while the feature tables
    store ``<number>.nii.gz``.  Both spellings must resolve.
    """
    if not isinstance(v, str):
        return v
    v = v.strip()
    if v in mapping:
        return mapping[v]
    if v.endswith(".nii.gz"):
        base = v[:-7]
        if base in mapping:
            return mapping[base] + ".nii.gz"
    return v


def map_series(s, mapping: dict):
    """Replace identifier values; leave anything that is not an identifier alone."""
    return s.map(lambda v: map_value(v, mapping))


def apply_to_frame(df, mapping: dict, columns=ID_COLUMNS):
    """Replace identifiers in every present identifier column, in place.

    Handles both ``<number>`` and ``<number>.nii.gz`` spellings (see
    :func:`map_value`).
    """
    touched = []
    for c in columns:
        if c not in df.columns:
            continue
        s = df[c].astype(str)
        n = 0
        out = []
        for v in s:
            nv = map_value(v, mapping)
            if nv != v:
                n += 1
            out.append(nv)
        if n:
            df[c] = out
            touched.append((c, int(n)))
    return touched


def apply_to_file(path, mapping: dict, columns=ID_COLUMNS) -> list:
    """Rewrite one CSV in place, preserving column order and row order."""
    import pandas as pd

    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    touched = apply_to_frame(df, mapping, columns)
    if touched:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return touched
