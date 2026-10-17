#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 ``data/`` 里所有住院号替换为研究专用编号 ``S####``（一次性维护脚本）。

    python tools/pseudonymise_repo.py --dry-run
    python tools/pseudonymise_repo.py

先备份到 ``backups/pseudonymise_<时间戳>/``，再原地改写；输出逐文件报告。
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hrpca import pseudonym as PS  # noqa: E402

SOURCES = [(r"C:\O\Clinic.csv", "ID"), (REPO / "data" / "labels.csv", "ID")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mapping = PS.load_map(SOURCES)
    print(f"映射: {len(mapping)} 个标识符 → {min(mapping.values())}…{max(mapping.values())}")

    targets = sorted(p for p in (REPO / "data").rglob("*.csv"))
    print(f"扫描 {len(targets)} 个 CSV\n")

    plan = []
    for p in targets:
        with io.open(p, encoding="utf-8-sig", errors="ignore") as fh:
            head = next(csv.reader(fh), [])
        cols = [c for c in head if c.strip() in PS.ID_COLUMNS]
        if cols:
            plan.append((p, cols))

    if not plan:
        print("没有含标识符列的文件。")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = REPO / "backups" / f"pseudonymise_{stamp}"

    total = 0
    for p, cols in plan:
        import pandas as pd
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        touched = PS.apply_to_frame(df, mapping, cols)
        n = sum(h for _, h in touched)
        rel = p.relative_to(REPO)
        print(f"  {str(rel):<42} 列 {cols}  命中 {n}")
        if args.dry_run or n == 0:
            continue
        bdir.mkdir(parents=True, exist_ok=True)
        bp = bdir / rel
        bp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, bp)
        df.to_csv(p, index=False, encoding="utf-8-sig")
        total += n

    if args.dry_run:
        print("\n[干跑] 未改写任何文件。")
    else:
        print(f"\n已替换 {total} 个标识符值；备份在 {bdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
