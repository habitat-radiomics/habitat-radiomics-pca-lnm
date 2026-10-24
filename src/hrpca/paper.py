"""解析权威稿件，抽出所有需要复现的数字。

★ 每次都从磁盘上的权威文件重新解析，**绝不使用硬编码的历史值**
（教训：2026-10-16，我引用了 workspace 里的过期对照稿
  `lunwen3\\投稿版_Manuscript_FINAL_EN.md`，它含已废弃的 "Table 4"，导致 11 项误报）。

权威文件（**双源**）：
  正文        <project>\\05_源文件\\Manuscript_FINAL.md
  补充材料    <project>\\02_补充材料\\Supplementary_Material.md
  在线数据    <project>\\15_在线数据\\*.csv

补充材料**有意瘦身**（33 → 18 页）：正文只印 4 张表（S3、S3b、S3d、S20），
其余表格移到 ``15_在线数据/`` 的 CSV 供在线发布。因此：
  * PDF 里仍印的表 → 解析 ``Supplementary_Material.md``
  * 已外移的表     → 解析 ``online.py`` 读到的 CSV
只读，不修改项目根目录。
"""
from __future__ import annotations

import os

import re
from pathlib import Path

from . import online as ON

MANUSCRIPT = Path(os.environ.get("HRPCA_MANUSCRIPT", ""))
SUPPLEMENT = Path(os.environ.get("HRPCA_SUPPLEMENT", ""))

#: tables that the slimmed supplement still prints
PRINTED_TABLES = ("Table S3", "Table S3b", "Table S3d", "Table S20")

#: tables that moved to the online-data CSVs
EXTERNAL_TABLES = ("S1a", "S1b", "S2", "S3c", "S4", "S5", "S6", "S8", "S9", "S10",
                   "S11", "S12", "S13", "S15", "S16", "S17", "S18", "S19", "S21",
                   "S22")

COHORTS = ["train", "test1", "test2", "test_prospective"]
SUPP_TABLE_RE = re.compile(r"^##\s*\*\*(Table S[0-9]+[a-z]?)\.")

# main-text table captions, e.g. **Table 2.** Predictive performance ...
MAIN_TABLE_RE = re.compile(r"^\*\*(Table [0-9]+)\.\*\*")


def _read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main_text_tables() -> dict:
    """``{table_name: [line1, line2, ...]}`` for the main text."""
    lines = _read(MANUSCRIPT)
    out, cur, buf = {}, None, []
    for ln in lines:
        m = MAIN_TABLE_RE.match(ln.strip())
        if m:
            if cur:
                out[cur] = buf
            cur, buf = m.group(1), [ln]
        elif cur:
            if ln.strip().startswith("###") or ln.strip().startswith("**Fig"):
                out[cur] = buf
                cur, buf = None, []
            else:
                buf.append(ln)
    if cur:
        out[cur] = buf
    return out


def supp_tables() -> dict:
    """``{table_name: [line1, line2, ...]}`` for the supplement."""
    lines = _read(SUPPLEMENT)
    out, cur, buf = {}, None, []
    for ln in lines:
        m = SUPP_TABLE_RE.match(ln)
        if m:
            if cur:
                out[cur] = buf
            cur, buf = m.group(1), [ln]
        elif cur:
            out[cur] = buf if ln.startswith("#") else buf
            if ln.startswith("#"):
                cur, buf = None, []
            else:
                buf.append(ln)
    if cur:
        out[cur] = buf
    return out


def parse_md_table(block: list[str]) -> list[list[str]]:
    """Extract the first markdown table in *block* as a list of rows."""
    rows, started = [], False
    for ln in block:
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                started = True
                continue
            rows.append(cells)
            started = True
        elif started and rows:
            break
    return rows


def num(s: str):
    """Parse a published cell into a float, tolerating ``.336`` / ``<.001`` / ``−0.041``.

    Note ``.336`` (no leading zero) is common in this manuscript, so the regex
    must accept a bare decimal point.
    """
    if s is None:
        return None
    s = s.replace("−", "-").replace("–", "-").strip()
    s = s.replace("*", "").replace("+", "").replace("\u00a0", " ")
    if s.startswith("<"):
        return 0.0
    m = re.match(r"^-?(?:\d+(?:\.\d*)?|\.\d+)", s)
    return float(m.group(0)) if m else None


def is_lt(s: str) -> bool:
    """True when the published cell is written as '<something'."""
    return bool(s) and s.replace("*", "").strip().startswith("<")


# ---------------------------------------------------------------------------
#: tables the verifier needs, by source.  Used by :func:`preflight` to tell an
#: incomplete submission workspace apart from a numerical disagreement.
EXPECTED_SUPP_TABLES = PRINTED_TABLES
EXPECTED_ONLINE_TABLES = EXTERNAL_TABLES


class SourceIncomplete(RuntimeError):
    """A published source this verifier needs is unavailable.

    Raised instead of a bare ``KeyError`` so that a rebuild in progress (or a
    missing online-data file) is reported as an ENVIRONMENT problem rather than
    as a numerical failure.
    """


def preflight():
    """``(ok, problems)`` — are the sources this verifier needs all readable?"""
    problems = []
    have = supp_tables()
    for t in PRINTED_TABLES:
        if t not in have:
            problems.append(f"supplement is missing printed {t}")
    ok_on, missing = ON.preflight()
    if not ok_on:
        problems.append("online data missing: " + ", ".join(missing))
    return (not problems), problems


def online_table(table_id: str):
    """``(header, rows)`` of an externalised table from ``15_在线数据/``."""
    try:
        return ON.get(table_id, 0)
    except Exception as exc:
        raise SourceIncomplete(
            f"cannot read online-data table {table_id!r} from {ON.ONLINE_DIR}: {exc}"
        ) from exc


def printed_table(name: str):
    """Raw markdown lines of a table that the slimmed supplement still prints."""
    have = supp_tables()
    if name not in have:
        raise SourceIncomplete(
            f"the submitted supplement does not contain {name!r}; it prints "
            f"{', '.join(sorted(have))}.  A rebuild may be in progress."
        )
    return have[name]


def _rows_as_md(table_id: str) -> list:
    """Render an online-data table as markdown lines so the same parsers work."""
    header, rows = online_table(table_id)
    out = ["| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
    for r in rows:
        rr = list(r) + [""] * (len(header) - len(r))
        out.append("| " + " | ".join(rr[:len(header)]) + " |")
    return out


def table_s5_delong() -> dict:
    """Table S5 — {cohort: [vs Clinical, vs H4, vs H2, vs All-habitat]}."""
    rows = parse_md_table(_rows_as_md("S5"))
    labels = {"Training": "train", "Test 1": "test1", "Test 2": "test2",
              "Prospective": "test_prospective", "Pooled external": "pooled"}
    out = {}
    for r in rows[1:]:
        if r[0] in labels:
            out[labels[r[0]]] = [num(c) for c in r[1:5]]
    return out


def table_s10_incremental() -> dict:
    """Table S10 — per cohort dict of published values."""
    rows = parse_md_table(_rows_as_md("S10"))
    labels = {"Training": "train", "Test 1": "test1", "Test 2": "test2",
              "Prospective": "test_prospective",
              "Pooled external (n = 618)": "pooled", "Pooled external": "pooled"}
    out = {}
    for r in rows[1:]:
        if r[0] in labels:
            out[labels[r[0]]] = dict(auc_comb=num(r[1]), auc_clin=num(r[2]), dauc=num(r[3]),
                                     delong_p=num(r[4]), idi=num(r[5]), idi_p=num(r[6]),
                                     nri=num(r[7]), nri_lo=num(r[8].split("-")[0].split("–")[0]),
                                     nri_hi=num(r[8].split("-")[-1].split("–")[-1]))
    return out


def table_s9_calibration() -> dict:
    """Table S9 — {signature: [train, test1, test2, prospective]}."""
    rows = parse_md_table(_rows_as_md("S9"))
    out = {}
    for r in rows[1:]:
        raw = r[0].replace("*", "").strip()
        if raw.startswith("Clinical"):
            key = "Clinic"
        elif raw.startswith("All-habitat"):
            key = "Habitat"
        elif raw.startswith("Combined"):
            key = "Combined"
        elif re.match(r"^Habitat H[1-5]$", raw):
            key = "Habitat" + raw.split()[-1]
        else:
            continue
        out[key] = [num(c) for c in r[1:5]]
    return out


def table_s15_regression() -> dict:
    """Table S15 — {variable: dict(uni=…, uni_p=…, multi=…, multi_p=…)}."""
    rows = parse_md_table(_rows_as_md("S15"))
    out = {}
    for r in rows[1:]:
        name = r[0].split("(")[0].strip()
        key = {"Age": "Age", "PI-RADS": "PI_RADS", "PSA": "PSA", "ISUP grade group": "ISUP",
               "Target-ROI volume": "target_roi_volume", "Prostate volume": "prostate_volume",
               "PSA density": "PSAD"}.get(name)
        if not key:
            continue

        def trip(cell):
            m = re.match(r"^\s*(-?[\d.]+)\s*\(([-\d.]+)[–\-]([-\d.]+)\)", cell.replace("−", "-"))
            return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (None,) * 3

        out[key] = dict(uni=trip(r[1]), uni_p=num(r[2]), multi=trip(r[3]), multi_p=num(r[4]))
    return out


def table_s16a_clinical() -> dict:
    """Table S16 — first block: {comparator: {cohort: auc}}.

    The file stacks two tables: the AUC block, then a second header
    (``Comparator, Comparator AUC, Combined AUC, ΔAUC, 95% CI, P``) followed by
    the ΔAUC rows.  Those ΔAUC rows repeat two of the AUC row labels, so a flat
    pass over every row would overwrite the AUC entries (and emit a bogus
    ``Comparator`` key).  Stop at the first row whose AUC cells are not all
    numbers — that is the second header.
    """
    rows = parse_md_table(_rows_as_md("S16"))
    out = {}
    for r in rows[1:]:
        if len(r) < 6:
            continue
        vals = [num(c) for c in r[1:6]]
        if any(v is None for v in vals):
            break
        out[r[0].strip()] = dict(train=vals[0], test1=vals[1], test2=vals[2],
                                 test_prospective=vals[3], pooled=vals[4])
    return out


def table1_baseline() -> dict:
    """Table 1 — continuous rows, categorical rows and cohort sizes."""
    rows = parse_md_table(main_text_tables()["Table 1"])
    cont, cat, n = {}, {}, {}
    CONT = {"Age": "Age", "PSA": "PSA", "PSA density": "PSAD",
            "Target region volume": "target_roi_volume",
            "Prostate volume": "prostate_volume"}
    section = None
    for r in rows[1:]:
        head = r[0].split("(")[0].strip().rstrip("§*").strip()
        if head in CONT:
            cont[CONT[head]] = [r[1], r[2], r[3], r[4]]
            continue
        if head == "ISUP grade group":
            section = "ISUP"
            continue
        if head == "PI-RADS category":
            section = "PI_RADS"
            continue
        if head == "pN status":
            section = None
            continue
        if section and head in ("1", "2", "3", "4", "5") and r[1]:
            cat.setdefault(section, {})[int(head)] = [num(c) for c in r[1:5]]
    header = rows[0]
    for i, h in enumerate(header[1:5]):
        m = re.search(r"n\s*=\s*(\d+)", h)
        if m:
            n[COHORTS[i]] = int(m.group(1))
    return dict(continuous=cont, categorical=cat, n=n, header=header)


def table2_auc() -> dict:
    """Table 2 — {model label: [train, test1, test2, prospective]} (first AUC of each cell)."""
    rows = parse_md_table(main_text_tables()["Table 2"])
    out = {}
    for r in rows[1:]:
        out[r[0].strip()] = [num(c.split("(")[0]) for c in r[1:5]]
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("main-text tables:", list(main_text_tables()))
    print("supplement tables:", sorted(supp_tables()))
    print()
    print("Table S5 :", table_s5_delong())
    print("Table S10:", table_s10_incremental())
    print("Table S9 :", table_s9_calibration())
    print("Table S15:", table_s15_regression())
    print("Table S16a:", table_s16a_clinical())
    print("Table 1 n:", table1_baseline()["n"])
    print("Table 1 continuous:", table1_baseline()["continuous"])
    print("Table 1 categorical:", table1_baseline()["categorical"])
    print("Table 2:", table2_auc())
