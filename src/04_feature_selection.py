#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""04 — feature selection (near-zero variance -> collinearity -> LASSO).

The three gates, in the order used in the study (Table S8):

======  ==========================================================================
 Gate    Rule
======  ==========================================================================
 NZV     drop near-zero-variance features (``VarianceThreshold``, default 1e-6)
 COLL    drop features with |Pearson r| > 0.9 to an already-kept feature
 LASSO   L1 logistic regression, 10-fold cross-validation, lambda = 0.0233
======  ==========================================================================

An optional ``--icc-table`` may be supplied; ICC is not applied by default.

Usage
-----
    python src/04_feature_selection.py --matrix data/candidate_features.csv
    python src/04_feature_selection.py --matrix ... --icc-table icc.csv --lasso
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrpca import banner, load_config, log, paths, write_json  # noqa: E402

ICC_CUTOFF = 0.80
CORR_CUTOFF = 0.90
VARIANCE_THRESHOLD = 1e-6
LASSO_LAMBDA = 0.0233
N_RETAINED = 25


# ---------------------------------------------------------------------------
def icc_filter(matrix: pd.DataFrame, icc: pd.Series) -> list:
    return [c for c in matrix.columns if float(icc.get(c, 0.0)) >= ICC_CUTOFF]


def variance_filter(matrix: pd.DataFrame, threshold: float = VARIANCE_THRESHOLD) -> list:
    from sklearn.feature_selection import VarianceThreshold

    sel = VarianceThreshold(threshold=threshold).fit(matrix.values)
    return list(np.asarray(matrix.columns)[sel.get_support()])


def collinearity_filter(matrix: pd.DataFrame, cutoff: float = CORR_CUTOFF,
                        order: list | None = None) -> list:
    """Greedy: walk the columns and drop any feature correlated above *cutoff*
    with a feature that has already been kept.

    ``order`` sets the visiting order (default: column order, which mirrors the
    order in which the features were ranked by the upstream pipeline).
    """
    cols = [c for c in (order or list(matrix.columns)) if c in matrix.columns]
    corr = matrix[cols].corr().abs()
    kept: list = []
    for c in cols:
        if all(corr.loc[c, k] <= cutoff for k in kept):
            kept.append(c)
    return kept


def lasso_select(matrix: pd.DataFrame, y: np.ndarray, lam: float = LASSO_LAMBDA,
                 seed: int = 0) -> tuple:
    """L1-penalised logistic regression with 10-fold CV on the penalty strength.

    ``lam`` is the pre-specified penalty; the function also reports the value
    chosen by cross-validation so that the two can be compared.
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(matrix.values)
    X = scaler.transform(matrix.values)
    fixed = LogisticRegression(penalty="l1", C=1.0 / max(lam, 1e-9), solver="liblinear",
                               max_iter=20000, random_state=seed).fit(X, y)
    cv = LogisticRegressionCV(penalty="l1", solver="liblinear", cv=10, max_iter=20000,
                              random_state=seed, Cs=30).fit(X, y)
    coefs = pd.DataFrame({"feature": matrix.columns, "coef": fixed.coef_[0]})
    coefs["selected"] = coefs["coef"].abs() > 1e-8
    return coefs, float(cv.C_[0]) and (1.0 / float(cv.C_[0])), scaler


def habitat_of(feature: str) -> int:
    import re

    m = re.search(r"_h(\d)$", feature)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--matrix", default=None, help="candidate feature matrix CSV")
    ap.add_argument("--icc-table", default=None, help="per-feature ICC table (CSV)")
    ap.add_argument("--lasso", action="store_true", help="also run the LASSO gate")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    banner("04 · feature selection (NZV -> collinearity -> LASSO)")

    matrix_path = Path(args.matrix) if args.matrix else P["data"] / "candidate_features.csv"
    if not matrix_path.exists():
        log(f"  candidate matrix not found: {matrix_path}")
        log("  build it first:  python src/03_feature_extraction.py --candidates")
        return 2
    matrix = pd.read_csv(matrix_path, encoding="utf-8-sig", index_col=0)
    log(f"  candidate matrix: {matrix.shape[0]} patients x {matrix.shape[1]} features")
    matrix = matrix.fillna(matrix.median(numeric_only=True))

    steps = {}
    cols = list(matrix.columns)
    if args.icc_table:
        icc = pd.read_csv(args.icc_table, encoding="utf-8-sig").set_index("feature")["icc"]
        cols = icc_filter(matrix[cols], icc)
        steps["ICC >= %.2f" % ICC_CUTOFF] = len(cols)
        log(f"  after ICC  >= {ICC_CUTOFF}: {len(cols)}")
    else:
        steps["ICC"] = "not run (optional --icc-table not supplied)"
        log("  ICC not applied (optional --icc-table not supplied)")

    cols = variance_filter(matrix[cols])
    steps["near-zero variance"] = len(cols)
    log(f"  after near-zero variance: {len(cols)}")

    cols = collinearity_filter(matrix[cols])
    steps["|r| <= %.1f" % CORR_CUTOFF] = len(cols)
    log(f"  after collinearity |r| <= {CORR_CUTOFF}: {len(cols)}")

    if args.lasso:
        labels = pd.read_csv(P["data"] / "scores" / "train.csv", encoding="utf-8-sig")["label"]
        # the candidate matrix is indexed by patient ID, the scores file is not,
        # so use the labels from the high-precision export keyed by ID
        hi = pd.read_csv(P["data"] / "scores_highprec.csv", encoding="utf-8-sig", dtype={"ID": str})
        tr = hi[hi.group == "train"].set_index("ID")
        idx = [i for i in matrix.index if i in set(tr.index)]
        coefs, cv_lambda, _ = lasso_select(matrix.loc[idx, cols], tr.loc[idx, "label"].values)
        kept = coefs[coefs["selected"]]
        steps["LASSO"] = len(kept)
        log(f"\n  LASSO with lambda = {LASSO_LAMBDA}: {len(kept)} features retained")
        log(f"  penalty chosen by 10-fold CV: {cv_lambda:.4f}")
        if len(kept) != N_RETAINED:
            log(f"  !! the paper reports {N_RETAINED} retained features — see "
                "docs/VERIFICATION_REPORT.md finding F8")
        coefs.to_csv(P["results_tables"] / "feature_selection_lasso.csv", index=False,
                     encoding="utf-8-sig")
        counts = pd.Series([habitat_of(f) for f in kept.feature]).value_counts().sort_index()
        log(f"  habitat distribution of retained features: {counts.to_dict()}")
        log(f"  paper reports: { {1: 5, 2: 7, 3: 7, 4: 5, 5: 3} }")
    else:
        steps["LASSO"] = "not run (pass --lasso)"

    write_json({"steps": steps,
                "cutoffs": {"icc": ICC_CUTOFF, "corr": CORR_CUTOFF,
                            "variance": VARIANCE_THRESHOLD, "lambda": LASSO_LAMBDA},
                "n_retained_paper": N_RETAINED},
               P["results"] / "feature_selection.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
