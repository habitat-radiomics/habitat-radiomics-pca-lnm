#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify the 22 externalised tables against what ``src/hrpca/paper.py`` parses.

Why this entry exists
---------------------
``15_在线数据/*.csv`` is now the primary source for 20 of the 22 supplementary
tables; only S3, S3b, S3d and S20 are still printed in the supplement.  Every
published number in ``check_paper_numbers.py`` therefore rests on the CSV parse
being faithful.  When the tables were restructured (S8 lost Panel B, S9 gained a
notes row, S1a gained a row, S1b was split in two, S16/S19 renamed a column)
that parse broke *silently* — only the check counts changed.  This gate makes
such a break loud.

Four independent checks
-----------------------
A  structure      every table parses into the block labels and row counts
                  declared in ``online.STRUCTURE``
B  accessor       each ``paper.py`` accessor reproduces an **independent**
                  positional read of the raw CSV, cell for cell
C  well-formed    every numeric-looking cell in every block of every table
                  parses to a finite number
D  cross-table    quantities stated in more than one table agree (pooled n,
                  pooled AUCs) — catches drift between tables generated
                  separately

Read-only: the submission package (``C:\\O\\投稿（压缩）``) is opened for reading;
nothing is copied into the repository.

Exit codes
    0  all checks pass
    1  a mismatch
    3  the submission package is unavailable (environment, not a result)
"""
from __future__ import annotations

import argparse
import math
import re
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrpca import online as ON          # noqa: E402
from hrpca import paper as PP           # noqa: E402
from hrpca import load_config, paths    # noqa: E402

RESULTS: list[dict] = []

COH_LABELS = {"Training": "train", "Test 1": "test1", "Test 2": "test2",
              "Prospective": "test_prospective", "Pooled external": "pooled",
              "Pooled external (n = 618)": "pooled",
              "Pooled validation": "pooled", "Pooled validation (n = 618)": "pooled"}
SIG_KEYS = {"Clinical": "Clinic", "All-habitat": "Habitat", "Combined": "Combined"}
S15_KEYS = {"Age": "Age", "PI-RADS": "PI_RADS", "PSA": "PSA", "ISUP grade group": "ISUP",
            "Target-ROI volume": "target_roi_volume", "Prostate volume": "prostate_volume",
            "PSA density": "PSAD"}


def check(section: str, item: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append(dict(section=section, item=item, verdict="PASS" if ok else "FAIL",
                        detail=detail))
    print(f"{'  =' if ok else '  X'} {section:<12} {item:<60} {detail}")
    return ok


def num(s):
    return PP.num(s)


def flat(tid: str) -> list:
    """Every data row of every block, in file order."""
    out = []
    for _, (_, rows) in ON.panels(tid).items():
        out.extend(rows)
    return out


def is_ascii_number(s: str) -> bool:
    t = (s.replace("−", "-").replace("–", "-").replace("<", "").replace(">", "")
         .replace("+", "").replace(",", "").replace("%", "").replace(".", "", 1))
    return bool(t) and all(ch in string.digits for ch in t) and len(t) < 12


# ---------------------------------------------------------------------------
# A · structure
# ---------------------------------------------------------------------------
def section_a() -> None:
    print("\n[A] 结构：每个表的块标签与行数 vs online.STRUCTURE")
    for tid, (labels, counts) in ON.STRUCTURE.items():
        try:
            ps = ON.panels(tid)
            got = [len(r) for _, (_, r) in ps.items()]
            glabels = list(ps.keys())
            ok = got == list(counts) and glabels == list(labels)
            check("A structure", f"{tid}: blocks={labels} rows={counts}", ok,
                  "" if ok else f"got blocks={glabels} rows={got}")
        except FileNotFoundError:
            check("A structure", f"{tid}: CSV present", False, "file missing")
        except ON.StructureError as exc:
            check("A structure", f"{tid}: parses", False, str(exc)[:70])


# ---------------------------------------------------------------------------
# B · accessor fidelity
# ---------------------------------------------------------------------------
def _eq(a, b, tol=1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _all_eq(got: dict, want: dict) -> bool:
    if set(got) != set(want):
        return False
    for k in want:
        if isinstance(want[k], dict):
            if set(got[k]) != set(want[k]):
                return False
            if not all(_eq(got[k][f], want[k][f]) for f in want[k]):
                return False
        else:
            if len(got[k]) != len(want[k]):
                return False
            if not all(_eq(x, y) for x, y in zip(got[k], want[k])):
                return False
    return True


def section_b() -> None:
    print("\n[B] 访问器保真：paper.py 的解析结果 vs 对原始 CSV 的独立位置读取")

    # ---- S5 : {cohort: [4 p values]}
    want = {}
    for r in flat("S5"):
        if r and r[0].strip() in COH_LABELS and len(r) >= 5:
            want[COH_LABELS[r[0].strip()]] = [num(c) for c in r[1:5]]
    got = PP.table_s5_delong()
    check("B accessor", f"S5 → table_s5_delong  ({len(want)} cohorts)", _all_eq(got, want))

    # ---- S9 : {signature: [4 H-L P]}
    want = {}
    for r in flat("S9"):
        raw = r[0].replace("*", "").strip()
        key = next((v for k, v in SIG_KEYS.items() if raw.startswith(k)), None)
        if key is None:
            m = re.match(r"^Habitat\s+H([1-5])$", raw)
            if m:
                key = "HabitatH" + m.group(1)      # paper.py spells it "HabitatH1"
        if key is None or len(r) < 5:
            continue
        want[key] = [num(c) for c in r[1:5]]
    got = PP.table_s9_calibration()
    check("B accessor", f"S9 → table_s9_calibration  ({len(want)} signatures)",
          _all_eq(got, want),
          "" if set(got) == set(want) else f"keys got={sorted(got)} want={sorted(want)}")

    # ---- S10
    want = {}
    for r in flat("S10"):
        if not r or r[0].strip() not in COH_LABELS or len(r) < 9:
            continue
        ci = r[8].replace("–", "-").replace("−", "-").split("-")
        want[COH_LABELS[r[0].strip()]] = dict(
            auc_comb=num(r[1]), auc_clin=num(r[2]), dauc=num(r[3]), delong_p=num(r[4]),
            idi=num(r[5]), idi_p=num(r[6]), nri=num(r[7]),
            nri_lo=num(ci[0]), nri_hi=num(ci[-1]))
    got = PP.table_s10_incremental()
    check("B accessor", f"S10 → table_s10_incremental  ({len(want)} cohorts)",
          _all_eq(got, want))

    # ---- S16a : must be the FIRST block only
    first = []
    for r in flat("S16"):
        if len(r) < 6:
            continue
        v = [num(c) for c in r[1:6]]
        if any(x is None for x in v):
            break                      # the second block header
        first.append((r[0].strip(), v))
    want = {n: dict(train=v[0], test1=v[1], test2=v[2],
                    test_prospective=v[3], pooled=v[4]) for n, v in first}
    got = PP.table_s16a_clinical()
    check("B accessor", f"S16 → table_s16a_clinical  ({len(want)} rows, block 1 only)",
          _all_eq(got, want),
          "" if set(got) == set(want) else f"keys got={sorted(got)} want={sorted(want)}")

    # ---- S15 : keys + every returned OR/CI occurs in that variable's row
    want_keys, bad = set(), []
    for r in flat("S15"):
        key = S15_KEYS.get(r[0].split("(")[0].strip())
        if not key:
            continue
        want_keys.add(key)
        tokens = set()
        for c in r:
            for tok in re.split(r"[\s()（）,;]+", str(c).replace("−", "-").replace("–", "-")):
                v = num(tok)
                if v is not None:
                    tokens.add(round(v, 4))
        for field in ("uni", "multi"):
            for v in got.get(key, {}).get(field, ()):
                if v is None:
                    continue
                if not any(abs(round(v, 4) - t) < 2e-3 for t in tokens):
                    bad.append(f"{key}.{field}={v}")
    got = PP.table_s15_regression()
    check("B accessor", f"S15 → table_s15_regression  ({len(want_keys)} variables)",
          set(got) == want_keys and not bad,
          "" if not bad else "not in row: " + ", ".join(bad[:4]))


# ---------------------------------------------------------------------------
# C · well-formed numeric cells (all blocks)
# ---------------------------------------------------------------------------
def section_c() -> None:
    print("\n[C] 数值完整性：每张表每个块的数字格都能解析为有限数")
    total = bad_total = 0
    for tid in ON.FILES:
        try:
            rows = flat(tid)
        except (FileNotFoundError, ON.StructureError) as exc:
            check("C numeric", f"{tid}: readable", False, str(exc)[:60])
            continue
        n_num, bad = 0, []
        for ri, r in enumerate(rows):
            for ci, cell in enumerate(r):
                s = str(cell).strip()
                if not s or not is_ascii_number(s):
                    continue
                n_num += 1
                v = num(s)
                if v is None or not math.isfinite(v):
                    bad.append(f"r{ri}c{ci}={s!r}")
        total += n_num
        bad_total += len(bad)
        check("C numeric", f"{tid}: {n_num} numeric cells parse", not bad,
              "" if not bad else ", ".join(bad[:3]))
    print(f"\n    合计 {total} 个数字格，{bad_total} 个不可解析")


# ---------------------------------------------------------------------------
# D · cross-table consistency
# ---------------------------------------------------------------------------
def section_d() -> None:
    print("\n[D] 跨表一致性：同一个量在多张表里必须相等")
    blocks = {}
    for tid in ("S10", "S16", "S17", "S19", "S20"):
        try:
            blocks[tid] = ON.panels(tid)
        except Exception:
            blocks[tid] = {}

    def head(tid, i=0):
        ps = blocks.get(tid) or {}
        keys = list(ps)
        return ps[keys[i]][0] if i < len(keys) else []

    def rows(tid, i=0):
        ps = blocks.get(tid) or {}
        keys = list(ps)
        return ps[keys[i]][1] if i < len(keys) else []

    def find(rows_, label, col):
        for r in rows_:
            if r and r[0].strip() == label and len(r) > col:
                return num(r[col])
        return None

    # ---- pooled n = 618
    stated = []
    for tid in ("S17", "S19", "S20"):
        for c in head(tid):
            if "pooled" not in c.lower():
                continue                       # S19's header also states 398/339/189/90
            m = re.search(r"\(n\s*=\s*([\d,]+)\)", c)
            if m:
                stated.append((tid, int(m.group(1).replace(",", ""))))
    for r in rows("S10"):
        if r and "n = 618" in r[0]:
            stated.append(("S10", 618))
    n17 = find(rows("S17"), "Pooled external", 1)
    if n17 is not None:
        stated.append(("S17A", int(n17)))
    ok = bool(stated) and all(v == 618 for _, v in stated)
    check("D cross", f"pooled n = 618 in {len(stated)} places", ok,
          ", ".join(f"{k}={v}" for k, v in stated))

    # ---- combined pooled AUC: S16 ΔAUC block vs S17A vs S20 panel 2
    s16 = rows("S16")
    a16 = None
    for i, r in enumerate(s16):
        if r and r[0].strip() == "Comparator":          # second-block header
            for r2 in s16[i + 1:]:
                if len(r2) > 2:
                    a16 = num(r2[2])                     # "Combined AUC"
                    break
            break
    a17 = find(rows("S17"), "Pooled external", 2)
    a20 = find(rows("S20", 1), "Combined", 1)
    trio = [(k, v) for k, v in (("S16", a16), ("S17A", a17), ("S20b", a20)) if v]
    if len(trio) >= 2:
        spread = max(v for _, v in trio) - min(v for _, v in trio)
        check("D cross",
              "combined pooled AUC: " + ", ".join(f"{k}={v:.4f}" for k, v in trio),
              spread <= 1e-3, f"spread={spread:.4f}")

    # ---- clinical pooled AUC: S16 AUC block vs S20b
    c16 = None
    for r in s16:
        if r and "PI-RADS" in r[0] and "primary" in r[0] and len(r) > 5:
            v = num(r[5])
            if v is not None:
                c16 = v
                break
    c20 = find(rows("S20", 1), "Clinical", 1)
    if c16 is not None and c20 is not None:
        check("D cross", f"clinical pooled AUC: S16={c16:.4f}, S20b={c20:.4f}",
              abs(c16 - c20) <= 1e-3, f"diff={abs(c16 - c20):.4f}")

    # ---- all-habitat pooled AUC: S20b vs S16 if the row exists
    h20 = find(rows("S20", 1), "All-habitat", 1)
    check("D cross", "S20b all-habitat pooled AUC present", h20 is not None, f"{h20}")

    # ---- S20 panel 2 rows: ΔAUC must equal the difference of the two columns
    bad = []
    for r in rows("S20", 1):
        if len(r) < 4:
            continue
        a, b, d = num(r[1]), num(r[2]), num(r[3])
        if None in (a, b, d):
            continue
        if abs((b - a) - d) > 5e-4:
            bad.append(f"{r[0].strip()}: {b}-{a}={b - a:+.4f} vs {d:+.4f}")
    check("D cross", f"S20b ΔAUC = Excluding − All ({len(rows('S20', 1))} rows)",
          not bad, "" if not bad else "; ".join(bad[:3]))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--gate", action="store_true", default=True)
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    args = ap.parse_args()

    banner = "=" * 100
    print(banner)
    print("  check_online_vs_paper.py —— 15_在线数据 的 22 张表 vs paper.py 的解析")
    print(f"  online dir   : {ON.ONLINE_DIR}")
    print(f"  supplement   : {PP.SUPPLEMENT}")
    print(f"  manuscript   : {PP.MANUSCRIPT}")
    print(banner)

    missing = []
    if not ON.ONLINE_DIR.is_dir():
        missing.append(f"online dir {ON.ONLINE_DIR}")
    else:
        found = {p.stem for p in ON.ONLINE_DIR.glob("*.csv")}
        absent = [v for v in ON.FILES.values() if v not in found]
        if absent:
            missing.append(f"{len(absent)} CSV(s) missing, e.g. {absent[:3]}")
    if not PP.SUPPLEMENT.is_file():
        missing.append(f"supplement {PP.SUPPLEMENT}")
    if missing:
        print(banner + "\n  PRECHECK FAILED — the submission package is not readable\n" + banner)
        for m in missing:
            print(f"    missing: {m}")
        print("\n  Environment problem, not a numerical one; exiting 3 so that")
        print("  automation never treats it as a pass.")
        print(banner)
        return 3

    section_a()
    section_b()
    section_c()
    section_d()

    df = pd.DataFrame(RESULTS)
    df.insert(0, "no", range(1, len(df) + 1))
    try:
        out = paths(load_config(args.config))["results_tables"] / "online_vs_paper.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        dest = str(out)
    except Exception as exc:
        dest = f"(not written: {type(exc).__name__}: {exc})"
    n_pass = int((df.verdict == "PASS").sum())
    n_fail = int((df.verdict == "FAIL").sum())
    print(f"\n{banner}\n  summary\n{banner}")
    print(f"  PASS {n_pass:>4}\n  FAIL {n_fail:>4}\n  full table -> {dest}")
    print(banner)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
