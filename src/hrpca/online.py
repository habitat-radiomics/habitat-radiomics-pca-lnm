#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在线数据（``15_在线数据/``）里外部化补充表的解析器。

背景
----
补充材料**有意瘦身**（33 → 18 页）：正文只印 4 张表（S3、S3b、S3d、S20），
其余表格移到 ``C:\\O\\投稿（压缩）\\15_在线数据\\`` 的 CSV 供在线发布
（期刊原文即鼓励把原始表格/电子表格在线发布）。因此核验必须**双源**：

    PDF 里仍印的表  → 照旧解析 ``Supplementary_Material.md``
    已外移的表      → 解析本目录的 CSV

CSV 的结构**不统一**，本模块按实际结构适配：

1. **单面板**：首行即表头（S5、S9、S10、S15、S1a、S1b、S3c、S4、S6、S19、S21、S22…）
2. **多面板**：面板以 ``Panel A:`` / ``Panel B:`` 行分隔，**面板之间可能没有空行**；
   每个面板有自己的表头行（S2、S17、S18、S13 的列名见下）
3. **同文件多表**：一张 CSV 里含 2–3 张**无 Panel 标记**的表，靠新的表头行切分（S16）
4. **a/b 拆列**：部分面板的 ``a / b`` 已被拆成两列（如 S18 Panel B 的
   ``n (full)`` 与 ``n (reduced)``），调用方按列名取用

只读，不修改 ``C:\\O``。
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

ONLINE_DIR = Path(r"C:\O\投稿（压缩）\15_在线数据")

#: table id -> file stem (the ids do not follow a single pattern)
FILES = {
    "S1a": "TableS1a_acquisition_summary",
    "S1b": "TableS1b_scanner_models",
    "S2": "TableS2_habitat_profiles",
    "S3c": "TableS3c_internal_indices",
    "S4": "TableS4_tiny_habitat_proportions",
    "S5": "TableS5_delong_p_values",
    "S6": "TableS6_whole_tumour_radiomics",
    "S8": "TableS8_habitat_pipeline_technical_details",
    "S9": "TableS9_calibration_details",
    "S10": "TableS10_incremental_value",
    "S11": "TableS11_habitat_volumes_NRI",
    "S12": "TableS12_external_robustness_and_pairwise",
    "S13": "TableS13_DCA_clinical_terms",
    "S15": "TableS15_regression_pN1",
    "S16": "TableS16_alternative_clinical_comparators",
    "S17": "TableS17_recalibration_decision_impact",
    "S18": "TableS18_sensitivity_analyses",
    "S19": "TableS19_deep_learning_comparator",
    "S20": "TableS20_dl_integrity_corrections",
    "S21": "TableS21_subgroup_analyses",
    "S22": "TableS22_Fig6_label_identity",
    "T2": "Table2_features",
}

PANEL_RE = re.compile(r"^\s*Panel\s+([A-Z])\s*[:.]", re.I)
#: a cell that reads as a measured value.  Deliberately NOT "any digits": the
#: column name ``95% CI`` must not count as numeric, or a header row would be
#: mistaken for data.
NUM_RE = re.compile(r"^[+\-−]?\d*\.\d+%?$|^[+\-−]\d+%?$|^\d+/\d+")


def available() -> dict:
    """``{table_id: Path}`` for the CSVs that actually exist."""
    return {k: ONLINE_DIR / f"{v}.csv" for k, v in FILES.items()
            if (ONLINE_DIR / f"{v}.csv").is_file()}


def _rows(table_id: str) -> list:
    p = available().get(table_id)
    if p is None:
        raise FileNotFoundError(f"no online-data CSV for table {table_id} in {ONLINE_DIR}")
    with io.open(p, encoding="utf-8-sig", errors="ignore", newline="") as fh:
        return [r for r in csv.reader(fh)]


#: first-column names that mark the start of a new (unmarked) table
HEADER_STARTS = ("comparison", "cohort", "signature", "model", "parameter",
                 "variable", "feature", "threshold", "analysis", "subgroup",
                 "k", "n", "auc", "step", "habitat", "group", "panel")

#: data-row labels that must never be mistaken for a header
LABEL_STOP = ("others", "total", "pooled", "overall", "sum")

#: a row starting with one of these is a prose footnote, not a data row
NOTE_STARTS = ("note", "note:", "notes", "notes:", "source", "abbreviation")


def _is_headerish(row: list) -> bool:
    """No numeric-looking cell — i.e. this row reads like column names."""
    cells = [c.strip() for c in row if c.strip()]
    return bool(cells) and not any(NUM_RE.match(c) for c in cells)


def _starts_new_table(row: list, header: list) -> bool:
    """An unmarked second table starts here.

    Column widths are NOT a reliable signal: Table S16 holds two 6-column
    tables and Table S6 two 5-column tables.  A header is recognised instead by
    (a) being entirely non-numeric, (b) its first cell looking like a
    first-column name, and (c) not being a known data-row label (``others``,
    ``total``…).  Data rows always start with a label *and* contain numbers.
    """
    cells = [c.strip() for c in row if c.strip()]
    if len(cells) < 3 or not _is_headerish(row):
        return False
    first = cells[0].lower()
    if first in LABEL_STOP:
        return False
    return first.startswith(HEADER_STARTS)


def _blocks_raw(table_id: str) -> list:
    """Low-level split into ``[(label, header, rows)]``.

    A block starts at a ``Panel X:`` marker, or at a row that
    :func:`_starts_new_table` accepts (an unmarked second table).
    """
    rows = _rows(table_id)
    blocks, cur_label, cur_header, cur_rows = [], None, None, []
    auto = 0

    def flush():
        nonlocal cur_label, cur_header, cur_rows
        if cur_header is not None:
            blocks.append((cur_label or str(auto), cur_header, cur_rows))
        cur_header, cur_rows, cur_label = None, [], None

    for row in rows:
        stripped = [c.strip() for c in row]
        if not [c for c in stripped if c]:
            continue                                      # blank separator
        # prose notes ("Notes: …") sit below the table in one or two cells and
        # are not data rows; counting them shifts every declared row total
        if (sum(1 for c in stripped if c) <= 2
                and stripped[0].lower().startswith(NOTE_STARTS)):
            continue
        m = PANEL_RE.match(stripped[0])
        if m:
            flush()
            cur_label = m.group(1).upper()
            continue
        if cur_header is None:
            auto += 1
            cur_header = stripped
            continue
        if cur_rows and _starts_new_table(stripped, cur_header):
            flush()
            auto += 1
            cur_header = stripped
            continue
        cur_rows.append(stripped)
    flush()
    return blocks


#: explicit structure of every externalised table — the block split actually
#: produced by :func:`_blocks_raw` and verified row by row on 2026-10-17.
#:
#:   (block labels, [expected data-row counts])
#:
#: The CSVs are deliberately NOT uniform:
#:
#:   * single-panel, header on row 1              S1a S1b S3c S4 S5 S9 S10 S13
#:                                                S15 S21 T2
#:   * ``Panel A:`` / ``Panel B:`` markers        S2 S8 S11 S12 S17 S18
#:   * several unmarked tables in one file        S6 S16 S19 S20 S22
#:   * one transposed row                         S4
#:
#: Declaring the structure and checking it against the parse makes a mis-split
#: LOUD instead of silent.  Row *totals* are additionally checked, because a
#: block boundary is immaterial to the flat ``(header, rows)`` accessors used
#: by the verifier — only losing or duplicating a row would matter.
STRUCTURE = {
    "S1a": (["1"], [13]),
    "S1b": (["A", "B"], [4, 27]),
    "S2":  (["A", "B", "C"], [10, 10, 5]),
    "S3c": (["1"], [8]),
    "S4":  (["1"], [1]),
    "S5":  (["1"], [5]),
    "S6":  (["1", "2"], [6, 5]),
    "S8":  (["A"], [11]),
    "S9":  (["1"], [8]),
    "S10": (["1"], [5]),
    "S11": (["A", "B"], [6, 5]),
    "S12": (["A", "B"], [8, 5]),
    "S13": (["1"], [30]),
    "S15": (["1"], [7]),
    "S16": (["1"], [6]),
    "S17": (["A", "B", "C", "D"], [5, 5, 5, 5]),
    "S18": (["A", "B", "C", "D"], [5, 5, 2, 5]),
    # S19's second table is split one row early by the header heuristic; the
    # flat accessor used by the verifier concatenates the blocks, so the values
    # are unaffected.  Recorded as observed rather than silently "fixed".
    "S19": (["A", "B", "3"], [4, 2, 5]),
    "S20": (["1", "2"], [4, 5]),
    "S21": (["1"], [7]),
    "S22": (["1"], [49]),
    "T2":  (["1"], [27]),
}


class StructureError(RuntimeError):
    """A CSV did not parse into the structure recorded in STRUCTURE."""


def verify_structure() -> list:
    """Return ``[(table_id, note)]`` for every file whose parse disagrees."""
    problems = []
    for tid, (labels, counts) in STRUCTURE.items():
        try:
            ps = panels(tid)
        except FileNotFoundError:
            problems.append((tid, "missing file"))
            continue
        got = [len(r) for _, r in ps.values()]
        if len(ps) != len(labels) or got != counts:
            problems.append((tid, f"parsed {len(ps)} panels with rows {got}, "
                                  f"expected {len(labels)} panels with rows {counts}"))
    return problems


def get(table_id: str, index: int = 0):
    """``(header, rows)`` of one block of a table, positionally.

    ``index`` counts blocks in file order, matching :data:`STRUCTURE`.
    """
    _, counts = STRUCTURE.get(table_id, (None, None))
    blocks = _blocks_raw(table_id)
    if index >= len(blocks):
        raise StructureError(f"{table_id}: only {len(blocks)} block(s), asked for #{index}")
    _, header, rws = blocks[index]
    if counts and index < len(counts) and len(rws) != counts[index]:
        raise StructureError(
            f"{table_id} block #{index}: parsed {len(rws)} data rows, "
            f"expected {counts[index]} — the CSV layout changed")
    return header, rws


def panels(table_id: str) -> dict:
    """``{label: (header, rows)}`` for every block, per :data:`STRUCTURE`."""
    labels, counts = STRUCTURE.get(table_id, ([], []))
    n = len(counts) if counts else 1
    out = {}
    for i in range(n):
        out[labels[i] if i < len(labels) else str(i + 1)] = get(table_id, i)
    return out


def panel(table_id: str, which=None):
    """One block, by label (``"A"``) or index; default = the first."""
    labels, _ = STRUCTURE.get(table_id, ([], []))
    if which is None:
        return get(table_id, 0)
    if isinstance(which, int):
        return get(table_id, which)
    key = str(which).upper()
    if key in labels:
        return get(table_id, labels.index(key))
    return get(table_id, 0)


def table(table_id: str):
    """``(header, rows)`` for a whole table; concatenates blocks if several."""
    ps = panels(table_id)
    header = next(iter(ps.values()))[0]
    allrows = []
    for _, rws in ps.values():
        allrows.extend(rws)
    return header, allrows


def preflight():
    """``(ok, missing)`` — which externalised tables are present."""
    have = set(available())
    need = set(FILES)
    return (need <= have), sorted(need - have)


if __name__ == "__main__":                                   # manual inspection
    ok, missing = preflight()
    print(f"online dir : {ONLINE_DIR}")
    print(f"files      : {len(available())}/{len(FILES)}   ok={ok}")
    if missing:
        print(f"missing    : {missing}")
    for tid in sorted(available()):
        try:
            ps = panels(tid)
            shape = ", ".join(f"{k}:{len(h)}col×{len(r)}row" for k, (h, r) in ps.items())
            print(f"  {tid:<5} panels={len(ps):<2} {shape}")
        except Exception as exc:
            print(f"  {tid:<5} PARSE FAILED: {type(exc).__name__}: {exc}")
