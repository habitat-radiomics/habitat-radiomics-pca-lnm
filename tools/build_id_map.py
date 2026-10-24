"""建立 ID → 研究专用编号 的映射表（**不得进入仓库**）。

规则
----
* 覆盖来源文件里出现的**全部**唯一 ID（研究队列 1016 + 筛选集其余 109 = 1125）
* 按**数值升序**排序后依次编号 `S0001`…`S1125`
  （确定性：任何人用同样的来源文件都能重建同一张表，便于复核）
* 输出到环境变量 ``HRPCA_ID_MAP_DIR`` 指定的目录 —— **仓库外**，
  且目录名与 README 都明确标注为受控材料

用法
----
    python tools/build_id_map.py            # 建立映射（幂等）
    python tools/build_id_map.py --verify   # 只校验，不写
"""
from __future__ import annotations

import os

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("HRPCA_ID_MAP_DIR", ""))
MAP_CSV = OUTDIR / "ID_MAP.csv"
MAP_JSON = OUTDIR / "id_map.json"

# 来源：这些文件里的 ID 列构成 ID 全域
SOURCES = [
    (os.environ.get("HRPCA_CLINIC_CSV", ""), "ID"),
    (REPO / "data" / "labels.csv", "ID"),
]
ID_RE = r"^\d{7,10}$"


def collect_ids() -> list:
    ids = set()
    for path, col in SOURCES:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"来源文件不存在: {p}")
        with io.open(p, encoding="utf-8-sig", errors="ignore") as fh:
            rd = csv.DictReader(fh)
            if col not in (rd.fieldnames or []):
                raise SystemExit(f"{p} 没有列 {col!r}；实际列: {rd.fieldnames}")
            for r in rd:
                v = (r.get(col) or "").strip()
                if v:
                    ids.add(v)
    return sorted(ids, key=int)


def build() -> dict:
    ids = collect_ids()
    if len(ids) > 9999:
        raise SystemExit(f"ID 数 {len(ids)} 超过 S0001–S9999 容量")
    m = {old: f"S{i:04d}" for i, old in enumerate(ids, start=1)}
    return m


def digest(m: dict) -> str:
    blob = json.dumps(m, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    m = build()
    print(f"唯一 ID 数: {len(m)}")
    print(f"编号范围  : {min(m.values())} … {max(m.values())}")
    print(f"映射指纹  : {digest(m)}")
    print("样例      : " + ", ".join(f"{k}→{v}" for k, v in list(m.items())[:5]))

    if args.verify:
        if MAP_JSON.exists():
            old = json.loads(MAP_JSON.read_text(encoding="utf-8"))["map"]
            same = old == m
            print(f"与已存映射一致: {same}")
            return 0 if same else 1
        print("尚无已存映射")
        return 0

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with io.open(MAP_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# 受控 —— 原始 ID ↔ 研究编号 映射表"])
        w.writerow(["# 本文件不得进入公开仓库；不得随稿件、补充材料或代码包提交"])
        w.writerow(["original_id", "study_id"])
        for k, v in m.items():
            w.writerow([k, v])
    MAP_JSON.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "n": len(m),
        "sha256_of_map": digest(m),
        "warning": "NEVER COMMIT — maps hospital admission numbers to study IDs",
        "rule": "sorted numerically, S0001..S%04d" % len(m),
        "map": m,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUTDIR / "!!_CONTROLLED_README.md").write_text(
        "# ⛔ 禁止上传 / 禁止提交\n\n"
        "本目录含 **院内住院号 ↔ 研究专用编号 的映射表**。\n\n"
        "| 文件 | 用途 |\n|---|---|\n"
        "| `ID_MAP.csv` | 人工核对用 |\n"
        "| `id_map.json` | 程序用（含 sha256 指纹） |\n\n"
        "**本目录位于代码仓库之外**（`<project parent directory>/` 根，"
        "不在 `habitat-radiomics-pca-lnm\\` 内），因此不会被 `git add`。\n\n"
        "请勿复制到仓库、补充材料或任何公开位置。\n"
        "若需共享数据，只共享已替换为 `S####` 的版本。\n",
        encoding="utf-8")
    print(f"\n已写:\n  {MAP_CSV}\n  {MAP_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
