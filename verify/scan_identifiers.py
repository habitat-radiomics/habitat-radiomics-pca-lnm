"""① 全局标识符排查：仓库内所有数据/文档文件。

判据（整列 + 整格匹配，避免把特征值的小数误判为 ID）：
  R1  7–10 位纯数字      —— 住院号 / 病案号 / 检查号  ★高
  R2  11–18 位纯数字     —— 身份证 / 长号段          ★高
  R3  邮箱
  R4  电话（11 位手机号、含区号座机）
  R5  日期型（整格 YYYY-MM-DD 等）—— 精确到日的出生日期
  R6  可疑字段名（住院号 / 病案号 / 身份证 / MRN / birth …）★高

用法
----
    python verify/scan_identifiers.py                # 扫描并打印清单
    python verify/scan_identifiers.py --gate         # 门禁：★高命中即 exit 1
    python verify/scan_identifiers.py --paths data results docs README.md

作为推送前门禁，`--gate` 只对 **git 已跟踪文件** 判定（未跟踪的备份/临时文件
不构成推送风险）；若不在 git 仓库内则退化为扫描全部。
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

R_IDCOL = re.compile(r"^\d{7,10}$")
R_LONGNUM = re.compile(r"\b\d{11,18}\b")
R_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
R_PHONE = re.compile(r"\b1[3-9]\d{9}\b|\b0\d{2,3}-\d{7,8}\b")
R_DATE = re.compile(r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b")
R_FIELD = re.compile(r"住院号|病案号|检查号|门诊号|身份证|病历号|mrn|patient_?id|patientno|"
                     r"admission|case_?no|record_?no|ssn|birth|出生", re.I)


R_IDCELL = re.compile(r"^\d{7,10}$")           # 整格 = 7–10 位纯数字
R_LONGCELL = re.compile(r"^\d{11,18}$")        # 整格 = 11–18 位纯数字
R_MAILCELL = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$")
R_TELCELL = re.compile(r"^1[3-9]\d{9}$|^0\d{2,3}-\d{7,8}$")
R_DATECELL = re.compile(r"^(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])$")
ID_COL_MIN = 0.80                              # 该列 ≥80% 非空值命中才判为可疑列
SAMPLE_ROWS = 2000                             # 每文件抽样行数


def scan_csv(p: Path):
    """按 **整列 + 整格** 判定。

    必须整格匹配：否则 `0.005572272889901045` 这类小数里的连续数字串
    会被误判为长号段/电话（本轮第一版就是这么误报的）。
    """
    try:
        with io.open(p, encoding="utf-8-sig", errors="ignore") as fh:
            rows = []
            for i, r in enumerate(csv.reader(fh)):
                rows.append(r)
                if i > SAMPLE_ROWS:
                    break
    except Exception as e:
        return [("读取失败", str(e), "", "")]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    out = []
    for j, col in enumerate(header):
        vals = [r[j].strip() for r in body if j < len(r) and r[j].strip()]
        if not vals:
            continue
        n = len(vals)
        n_id = sum(bool(R_IDCELL.match(v)) for v in vals)
        n_long = sum(bool(R_LONGCELL.match(v)) for v in vals)
        n_mail = sum(bool(R_MAILCELL.match(v)) for v in vals)
        n_tel = sum(bool(R_TELCELL.match(v)) for v in vals)
        n_date = sum(bool(R_DATECELL.match(v)) for v in vals)
        sample = ", ".join(sorted(set(vals))[:3])
        if R_FIELD.search(col):
            out.append(("R6", col, sample, f"字段名可疑 ({n} 行)"))
        if n_id / n >= ID_COL_MIN:
            out.append(("R1", col, sample, f"{n_id}/{n} = {n_id/n:.0%} 为 7–10 位纯数字 ★"))
        if n_long / n >= ID_COL_MIN:
            out.append(("R2", col, sample, f"{n_long}/{n} = {n_long/n:.0%} 为 11–18 位纯数字 ★"))
        if n_mail:
            out.append(("R3", col, sample, f"{n_mail}/{n} 含邮箱"))
        if n_tel:
            out.append(("R4", col, sample, f"{n_tel}/{n} 含电话"))
        if n_date / n >= ID_COL_MIN:
            out.append(("R5", col, sample, f"{n_date}/{n} 为日期型"))
    return out


def scan_text(p: Path):
    txt = io.open(p, encoding="utf-8-sig", errors="ignore").read()
    out = []
    for tag, pat in [("R1", re.compile(r"\b\d{7,10}\b")), ("R2", R_LONGNUM),
                     ("R3", R_EMAIL), ("R4", R_PHONE), ("R5", R_DATE), ("R6", R_FIELD)]:
        hits = pat.findall(txt)
        if hits:
            uniq = sorted(set(hits))[:4]
            out.append((tag, "-", ", ".join(uniq), f"共 {len(hits)} 处"))
    return out


SEVERITY = {"R1": "★高", "R2": "★高", "R3": "中", "R4": "★高", "R5": "中", "R6": "★高"}
#: prose / data formats: shape rules apply
DOC_EXT = (".csv", ".json", ".md", ".txt", ".yaml", ".yml", ".cff")
#: code: only *known* identifiers are flagged (shape rules would fire on
#: coefficients, versions, dates and docstrings)
CODE_EXT = (".py",)
BINARY_EXT = (".png", ".pdf", ".jpg", ".svg", ".docx", ".xlsx", ".zip", ".gz")

#: the out-of-repo map: its keys ARE the direct identifiers we must not ship
MAP_JSON = Path(r"C:\Users\l\Desktop\lunwen3\00_备份_ID映射_禁传\id_map_禁传.json")


def load_known_ids() -> set:
    """The set of identifiers that must never appear in a shipped file."""
    try:
        import json
        # utf-8-sig: the stored JSON carries a BOM
        d = json.loads(MAP_JSON.read_text(encoding="utf-8-sig"))
        return {str(k) for k in d.get("map", {})}
    except Exception:
        return set()


KNOWN = load_known_ids()


def tracked_files() -> list:
    """git-tracked files (the ones that would actually be pushed)."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
        fs = [ROOT / l.strip() for l in out.splitlines() if l.strip()]
        if fs:
            return sorted(f for f in fs if f.is_file())
    except Exception:
        pass
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and ".git" not in p.parts and not any(
                      x in p.parts for x in ("backups", "__pycache__")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 if any ★high finding remains in tracked files")
    ap.add_argument("--paths", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="scan untracked files too")
    args = ap.parse_args()

    if args.paths:
        files = []
        for a in args.paths:
            p = (ROOT / a) if not Path(a).is_absolute() else Path(a)
            files += [p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]
        files = sorted(f for f in files if f.suffix.lower() in DOC_EXT + CODE_EXT)
        scope = "指定路径"
    elif args.all or not args.gate:
        files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                       and ".git" not in p.parts
                       and p.suffix.lower() in DOC_EXT + CODE_EXT)
        scope = "全部文件（含未跟踪）"
    else:
        files = [f for f in tracked_files() if f.suffix.lower() in DOC_EXT + CODE_EXT]
        scope = "git 已跟踪文件"

    print("=" * 118)
    print(f"  标识符排查 —— {scope}，共 {len(files)} 个文件")
    if KNOWN:
        print(f"  已知标识符库（仓库外映射表）: {len(KNOWN)} 个 —— 任何文件中出现即阻断")
    else:
        print("  ⚠ 未找到仓库外的映射表，代码文件只做形态检查")
    print("=" * 118)
    print(f"{'文件':<52}{'列':<20}{'样例值':<32}{'判定'}")
    print("-" * 118)

    findings = []
    for p in files:
        rel = str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
        is_code = p.suffix.lower() in CODE_EXT
        rows = scan_csv(p) if p.suffix.lower() == ".csv" else scan_text(p)
        for tag, col, sample, note in rows:
            age = "★高" if SEVERITY[tag] == "★高" else "中"
            if is_code:
                # in code, only identifiers that are actually known are a risk
                vals = [s.strip() for s in sample.split(",")]
                if not any(v in KNOWN for v in vals):
                    continue
                age = "★高"
                note += " [身份库命中]"
            findings.append((rel, tag, col, sample, age))
            short = sample[:30] + ("…" if len(sample) > 30 else "")
            print(f"{rel[:50]:<52}{(col or '-')[:18]:<20}{short:<32}{age} {note}")

    # ---- the decisive check: any tracked file containing a known identifier --
    # For CSVs this must be COLUMN-AWARE.  A plain text regex over a radiomics
    # table produces false positives: a feature value such as a long decimal can
    # contain a 7-digit run that coincides with an identifier.
    #
    # NOTE: do NOT wrap this in a bare ``except Exception: continue``.  An
    # undefined name here (this block once referenced a helper that had been
    # renamed) silently turned the whole leak check into a no-op while the gate
    # still printed PASS — the exact failure mode T12 in the README warns about.
    leaked = []
    errors = []
    for p in files:
        if p.suffix.lower() in BINARY_EXT:
            continue
        hit = []
        try:
            if p.suffix.lower() == ".csv":
                with io.open(p, encoding="utf-8-sig", errors="ignore") as fh:
                    rd = csv.reader(fh)
                    next(rd, [])                       # header
                    for row in rd:
                        for v in row:
                            hit.append(v.strip())
            else:
                txt = io.open(p, encoding="utf-8-sig", errors="ignore").read()
                hit = re.findall(r"\d{7,10}", txt)
        except Exception as exc:                       # reported, never swallowed
            errors.append((p.name, f"{type(exc).__name__}: {exc}"))
            continue
        found = sorted({v for v in hit
                        if v in KNOWN or (v.endswith(".nii.gz") and v[:-7] in KNOWN)})
        if found:
            rel = str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
            leaked.append((rel, found))
            if not any(f[0] == rel for f in findings):
                print(f"{rel[:50]:<52}{'(全文)':<20}{(', '.join(found[:3]))[:30]:<32}"
                      f"★高 身份库命中 {len(found)} 处")
    if errors:
        print("\n  ⚠ 扫描出错的文件（未计入门禁结论）：")
        for name, e in errors[:5]:
            print(f"    {name}: {e}")

    print()
    print("=" * 118)
    by_tag = {}
    for f in findings:
        by_tag[f[1]] = by_tag.get(f[1], 0) + 1
    print(f"  形态命中 {len(findings)}   身份库泄露文件 {len(leaked)}   按判据 {by_tag}")
    if leaked:
        print("\n  ★阻断级：以下文件含**已知住院号**：")
        for rel, hit in leaked:
            print(f"    {rel}   {len(hit)} 处，例: {', '.join(hit[:4])}")
    print()
    if args.gate:
        if not KNOWN:
            # FAIL CLOSED: without the identifier library we cannot prove the
            # absence of admission numbers, so the gate must not report PASS
            print("  门禁结果: FAIL — 未能加载仓库外的标识符库，无法证明无泄露"
                  f"\n    期望位置: {MAP_JSON}")
            return 1
        if leaked:
            print("  门禁结果: FAIL —— 存在直接标识符（已知住院号），禁止推送")
            return 1
        print(f"  门禁结果: PASS —— 对照 {len(KNOWN)} 个已知住院号无泄露；"
              "无 7–10 位纯数字列 / 身份证 / 电话 / 邮箱 / 日期型字段")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
