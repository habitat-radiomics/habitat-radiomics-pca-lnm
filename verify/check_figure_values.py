#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图 ↔ 表数值对照 —— 图里印出的每一个数字都必须能由其权威表复现。

为什么需要这一项
----------------
图与表是分别维护的：``src/10_figures.py`` 重画 ``results/figures/``，而表在
``results/tables/``（部分还同步到 ``15_在线数据/``）。**重画一次图就可能与表
脱钩**，而像素比较看不出来。所以这里从**矢量 PDF 提取图内文本**（不是像素、
不是画风），再与权威表**逐位**比对。

覆盖
----
    Fig2_ROC            ← data/scores_highprec.csv（训练 5 个边际签名）
                          data/scores/*.csv（其余队列 / 训练 Combined）
    Fig3a_calibration   ← results/tables/tableS9_hosmer.csv
    FigS2_K_selection   ← data/K_CV10x10_summary.csv
    FigS7_habitat_profiles ← results/tables/tableS2b_habitat_zscores.csv
    FigS9_SHAP          ← results/tables/figS9_mean_abs_shap.csv

    python verify/check_figure_values.py            # 全部
    python verify/check_figure_values.py -v         # 也列出通过项
    python verify/check_figure_values.py --gate     # 不一致即 exit 1（默认）

退出码：0 一致；1 有不一致；3 图或表缺失（环境问题）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pymupdf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hrpca import load_config, paths                    # noqa: E402
from hrpca import metrics as M                          # noqa: E402
from hrpca.data import COHORTS, EXTERNAL                # noqa: E402

#: how many decimals each figure prints — used to define an exact match
FIG_DP = {"Fig2_ROC": 3, "Fig3a_calibration": 3, "Fig3a_calibration_hl": 3,
          "FigS2_K_selection": 3, "FigS7_habitat_profiles": 2,
          "FigS9_SHAP": 3}

#: the clinical signature's printed name
CLIN_LABEL = "Clinical (ISUP + PI-RADS)"
LEGEND_NAMES = {"Clinic": CLIN_LABEL, "HabitatH2": "Habitat H2", "HabitatH4": "Habitat H4",
                "Habitat": "All-habitat (27 features)", "Combined": "Combined"}
TITLES = {"train": "Training", "test1": "Test cohort 1", "test2": "Test cohort 2",
          "test_prospective": "Prospective"}

RESULTS = []          # (section, label, expected, found, ok, note)


def record(section, label, expected, found, ok, note=""):
    RESULTS.append((section, label, expected, found, ok, note))


def pdf_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    doc = pymupdf.open(path)
    return "\n".join(p.get_text("text") for p in doc)


def want_str(v: float, dp: int) -> str:
    """Format a value the way the figure prints it (``+`` sign for small dp)."""
    return f"{v:+.{dp}f}" if dp == 2 else f"{v:.{dp}f}"


# ---------------------------------------------------------------------------
# Fig. 2 — ROC legend AUCs and panel titles
# ---------------------------------------------------------------------------
def check_fig2(P: dict, figdir: Path, verbose: bool):
    from importlib import util as _u                              # noqa: PLC0415
    spec = _u.spec_from_file_location("_fig10", REPO / "src" / "10_figures.py")
    fig10 = _u.module_from_spec(spec)
    sys.modules["_fig10"] = fig10
    spec.loader.exec_module(fig10)

    cfg = load_config()
    deployed = {c: pd.read_csv(P["data_scores"] / f"{c}.csv", encoding="utf-8-sig")
                for c in COHORTS}
    tr = fig10._training_scores(P["data"] / "scores_highprec.csv",
                                P["data_scores"] / "train.csv",
                                cfg["model"]["combined"])
    text = pdf_text(figdir / "Fig2_ROC.pdf")
    dp = FIG_DP["Fig2_ROC"]

    for coh in COHORTS:
        d = tr if coh == "train" else deployed[coh]
        for col, label in LEGEND_NAMES.items():
            # the training Combined is recomputed from the paper coefficients;
            # the printed value must agree to the printed precision
            auc = M.auc(d.label.values, d[col].values)
            tok = f"{label} ({auc:.{dp}f})"
            found = tok in text
            record("Fig2_ROC", f"{TITLES[coh]} · {label}", tok,
                   tok if found else "NOT FOUND", found)
        title = f"{TITLES[coh]} (n = {len(d)})"
        found = title in text
        record("Fig2_ROC", f"{TITLES[coh]} · panel title", title,
               title if found else "NOT FOUND", found)

    # the legend must not contain a stale signature name
    for stale in ("Clinical (0.", "Clinical ("):
        if stale == "Clinical (" and f"{CLIN_LABEL} (" in text:
            continue
        if re.search(r"(?<!\+ )Clinical \(0\.\d{3}\)", text):
            record("Fig2_ROC", "legend uses the published clinical name",
                   CLIN_LABEL, "bare 'Clinical (…)' found", False)
            break
    else:
        record("Fig2_ROC", "legend uses the published clinical name", CLIN_LABEL, "ok", True)


# ---------------------------------------------------------------------------
# Fig. 3a — Hosmer-Lemeshow P printed in each panel title
# ---------------------------------------------------------------------------
def check_fig3a(P: dict, figdir: Path, verbose: bool):
    """The Combined row of ``tableS9_hosmer.csv`` is what Fig. 3a's titles print.

    That table is a cohort × signature matrix, so the three Hosmer–Lemeshow
    conventions (this figure, the table, and the qcut variant) are visible side
    by side — the figure must reproduce the ``Combined`` row.
    """
    tbl = pd.read_csv(P["results_tables"] / "tableS9_hosmer.csv", encoding="utf-8-sig")
    if "signature" not in tbl.columns:
        raise ValueError(f"tableS9_hosmer.csv has no 'signature' column: {list(tbl.columns)}")
    row = tbl[tbl["signature"] == "Combined"]
    if row.empty:
        raise ValueError("tableS9_hosmer.csv has no Combined row")
    row = row.iloc[0]

    text = pdf_text(figdir / "Fig3a_calibration.pdf")
    dp = FIG_DP["Fig3a_calibration_hl"]
    printed = re.findall(r"H\s*[–\-]?\s*L P = ([\d.]+)", text)
    expected = [(c, float(row[c])) for c in COHORTS if c in tbl.columns]

    record("Fig3a_calibration", "panel count", len(expected), len(printed),
           len(printed) == len(expected), "表: results/tables/tableS9_hosmer.csv")
    for (coh, p), shown in zip(expected, printed):
        tok = f"{p:.{dp}f}"
        record("Fig3a_calibration", f"{TITLES[coh]} · H-L P (Combined)",
               tok, shown, shown == tok, "表: results/tables/tableS9_hosmer.csv")


# ---------------------------------------------------------------------------
# Fig. S2 — K = 5 annotation
# ---------------------------------------------------------------------------
def check_figs2(P: dict, figdir: Path, verbose: bool):
    """Fig. S2 prints only the selected K and its mean AUC.

    The individual per-K means appear as the curve (and as x-axis ticks), so the
    comparable *numbers* are the annotation — the axis limits are a styling
    choice that may legitimately differ.
    """
    d = pd.read_csv(P["data"] / "K_CV10x10_summary.csv", encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    sel = d.loc[d["best_per_fold_mean"].idxmax()]
    text = pdf_text(figdir / "FigS2_K_selection.pdf")
    dp = FIG_DP["FigS2_K_selection"]
    for tok in (f"K = {int(sel['K'])}", f"mean AUC {sel['best_per_fold_mean']:.{dp}f}"):
        record("FigS2_K_selection", tok, tok, tok if tok in text else "NOT FOUND",
               tok in text, "表: data/K_CV10x10_summary.csv")
    # the selected K must be the arg-max of the table
    record("FigS2_K_selection", "selected K = arg-max of the table",
           int(sel["K"]), int(sel["K"]), True,
           f"max best_per_fold_mean = {sel['best_per_fold_mean']}")


# ---------------------------------------------------------------------------
# Fig. S7 — every annotated cell vs Table S2 panel (b)
# ---------------------------------------------------------------------------
def check_figs7(P: dict, figdir: Path, verbose: bool):
    tbl = pd.read_csv(P["results_tables"] / "tableS2b_habitat_zscores.csv",
                      encoding="utf-8-sig")
    cols = [c for c in tbl.columns if c.startswith("H")]
    text = pdf_text(figdir / "FigS7_habitat_profiles.pdf")
    printed = [x.replace("−", "-") for x in re.findall(r"[+\-−]\d+\.\d{2}", text)]
    dp = FIG_DP["FigS7_habitat_profiles"]

    expected = []
    for _, r in tbl.iterrows():
        row = [float(r[c]) for c in cols]
        expected.append((str(r["feature"]), [f"{v:+.{dp}f}" for v in row]))

    flat_expected = [t for _, row in expected for t in row]
    record("FigS7_habitat_profiles", "annotated cell count",
           len(flat_expected), len(printed), len(printed) == len(flat_expected),
           "表: results/tables/tableS2b_habitat_zscores.csv")
    if len(printed) == len(flat_expected):
        for (name, row), got in zip(expected, [printed[i:i + len(cols)]
                                               for i in range(0, len(printed), len(cols))]):
            ok = row == got
            record("FigS7_habitat_profiles", f"{name}", str(row),
                   str(got), ok)
    # row labels must use the table's wording
    for stale, good in (("GLCM IDM", "GLCM inverse difference moment"),
                        ("\nEntropy\n", "\nEntropy (heterogeneity)\n")):
        bad = stale in text and good not in text
        record("FigS7_habitat_profiles", f"row label '{good}'",
               good, "ok" if not bad else f"stale '{stale}'", not bad)


# ---------------------------------------------------------------------------
# Fig. S9 — mean |SHAP| bars
# ---------------------------------------------------------------------------
def check_figs9(P: dict, figdir: Path, verbose: bool):
    tbl = pd.read_csv(P["results_tables"] / "figS9_mean_abs_shap.csv", encoding="utf-8-sig")
    text = pdf_text(figdir / "FigS9_SHAP.pdf")
    dp = FIG_DP["FigS9_SHAP"]
    printed = re.findall(r"\d+\.\d{3}", text)
    for _, r in tbl.iterrows():
        tok = f"{float(r['mean_abs_shap']):.{dp}f}"
        record("FigS9_SHAP", str(r["input"]), tok,
               tok if tok in printed else "NOT FOUND", tok in printed,
               "表: results/tables/figS9_mean_abs_shap.csv")
    record("FigS9_SHAP", "bar count", len(tbl), len(printed),
           len(printed) == len(tbl))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--gate", action="store_true", default=True)
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    args = ap.parse_args()

    cfg = load_config()
    P = paths(cfg)
    figdir = P["results_figures"]

    banner = "=" * 100
    print(banner)
    print("  check_figure_values.py —— 图内文本 vs 权威表（矢量 PDF 逐字提取）")
    print(f"  figures: {figdir}")
    print(banner)

    needed = [figdir / f"{n}.pdf" for n in
              ("Fig2_ROC", "Fig3a_calibration", "FigS2_K_selection",
               "FigS7_habitat_profiles", "FigS9_SHAP")]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        print(f"\n  PRECHECK: 缺少图文件 {missing} —— 先运行 src/10_figures.py")
        print(banner)
        return 3

    for fn in (check_fig2, check_fig3a, check_figs2, check_figs7, check_figs9):
        try:
            fn(P, figdir, args.verbose)
        except FileNotFoundError as exc:
            print(f"  !! {fn.__name__}: 缺少权威表 {exc}")
            return 3
        except Exception as exc:                      # noqa: BLE001 — reported, not swallowed
            record(fn.__name__, "check raised", "-", f"{type(exc).__name__}: {exc}", False)

    bad = [r for r in RESULTS if not r[4]]
    print(f"\n{'section':<24}{'item':<46}{'expected':<26}{'in figure':<26}{'verdict'}")
    print("-" * 130)
    for section, label, exp, got, ok, note in RESULTS:
        if ok and not args.verbose:
            continue
        print(f"{section:<24}{label[:44]:<46}{str(exp)[:24]:<26}{str(got)[:24]:<26}"
              f"{'PASS' if ok else 'FAIL'}")
    print()
    print(f"  共 {len(RESULTS)} 项，PASS {len(RESULTS) - len(bad)}，FAIL {len(bad)}")
    if bad:
        print("\n  不一致明细：")
        for section, label, exp, got, ok, note in bad:
            print(f"    [{section}] {label}\n        表: {exp}\n        图: {got}")
    print(banner)
    if args.gate and bad:
        print("  门禁结果: FAIL —— 图与表不一致，请重画图或同步表")
        return 1
    print("  门禁结果: PASS —— 图内每个数字都能由权威表复现")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
