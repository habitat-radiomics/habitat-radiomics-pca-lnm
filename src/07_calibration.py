#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""07 — calibration: Hosmer-Lemeshow, Cox recalibration, Brier score.

Reproduces **Table S9** (combined model), **Table S17(a)** and the calibration
Brier scores quoted in Results 3.5.

Convention (fixed, because the numbers are sensitive to it):

* equal-frequency binning with 10 bins (``pandas.qcut``, duplicates dropped);
* statistic ``sum (O-E)^2 / (E (1 - E/n))``, reference chi-square with
  ``bins - 2`` degrees of freedom;
* Cox recalibration fitted **within each cohort**:
  ``logit(p*) = a + b * logit(p)``.

Usage
-----
    python src/07_calibration.py
    python src/07_calibration.py --bins 10 --dump-bins
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrpca import banner, load_config, log, paths, write_json  # noqa: E402
from hrpca import metrics as M                                # noqa: E402
from hrpca.data import COHORTS, EXTERNAL, SIGNATURES          # noqa: E402


def hl_table(scores, pooled, bins: int) -> pd.DataFrame:
    rows = []
    for sig in SIGNATURES:
        row = {"signature": sig}
        for name, d in list(scores.items()) + [("pooled", pooled)]:
            row[name] = round(M.hosmer_lemeshow(d.label.values, d[sig].values, bins), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def recalibration_table(scores, pooled, bins: int) -> pd.DataFrame:
    rows = []
    for name, d in list(scores.items()) + [("pooled", pooled)]:
        y = d.label.values
        p = d.Combined.values
        a, b, pr = M.cox_recalibration(y, p)
        rows.append(dict(cohort=name, n=len(d),
                         auc=round(M.auc(y, p), 4),
                         intercept=round(a, 4), slope=round(b, 4),
                         hl_original=round(M.hosmer_lemeshow(y, p, bins), 4),
                         hl_recalibrated=round(M.hosmer_lemeshow(y, pr, bins), 4),
                         brier_original=round(M.brier(y, p), 4),
                         brier_recalibrated=round(M.brier(y, pr), 4)))
    return pd.DataFrame(rows)


def calibration_bins(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p, "bin": pd.qcut(p, bins, duplicates="drop")})
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"),
                                            predicted=("p", "mean"))
    g["observed"] = g["observed"].round(4)
    g["predicted"] = g["predicted"].round(4)
    return g.reset_index()


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--bins", type=int, default=None)
    ap.add_argument("--dump-bins", action="store_true", help="write per-bin calibration tables")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    bins = args.bins or cfg["statistics"]["hosmer_lemeshow_bins"]
    banner(f"07 · calibration (Hosmer-Lemeshow, {bins} equal-frequency bins, Cox recalibration)")

    scores = {c: pd.read_csv(P["data_scores"] / f"{c}.csv", encoding="utf-8-sig") for c in COHORTS}
    pooled = pd.concat([scores[c] for c in EXTERNAL], ignore_index=True)

    log("\n[1] Table S9 — H-L P value per signature and cohort")
    t9 = hl_table(scores, pooled, bins)
    log(t9.to_string(index=False))
    t9.to_csv(P["results_tables"] / "tableS9_hosmer.csv", index=False, encoding="utf-8-sig")

    log("\n[2] Table S17(a) — Cox recalibration of the combined model")
    t17 = recalibration_table(scores, pooled, bins)
    log(t17[["cohort", "n", "auc", "intercept", "slope", "hl_original",
             "hl_recalibrated"]].to_string(index=False))
    t17.to_csv(P["results_tables"] / "tableS17a_recalibration.csv", index=False,
               encoding="utf-8-sig")

    log("\n[3] Brier score (Results 3.5)")
    for _, r in t17.iterrows():
        log(f"  {r.cohort:<18} Brier {r.brier_original:.4f}")

    if args.dump_bins:
        for name, d in list(scores.items()) + [("pooled", pooled)]:
            b = calibration_bins(d.label.values, d.Combined.values, bins)
            b.to_csv(P["results_tables"] / f"calibration_bins_{name}.csv", index=False,
                     encoding="utf-8-sig")
        log("\n  per-bin calibration tables written")

    write_json({"bins": bins, "convention": "equal-frequency (qcut), df = bins - 2"},
               P["results"] / "calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
