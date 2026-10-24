#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""11 — habitat-number (K) selection by repeated nested cross-validation.

.. warning::

   **This is the K-selection flow, not the final-model flow.**

Four things distinguish it from the pipeline that produces the reported model
(``src/05_model_training.py`` … ``src/09_shap.py``):

1. **Purpose.**  It only answers "how many habitats (K = 2–9) should the
   partition have?" — it does not fit, calibrate or export the clinical model.
2. **Feature space.**  It runs on a *reduced* table: ``original_*`` image types
   only, i.e. **105 features per habitat** (K = 5 → 525 columns).  The final
   model uses all 13 image types (Original + 8 wavelet sub-bands + 4 LoG scales)
   — 1,197 features per habitat, 5,985 across five habitats.
3. **In-fold re-running.**  Standardisation and LASSO feature selection are
   re-fitted **inside every training fold**; the test fold never takes part in
   any fit.  ICC filtering, near-zero-variance removal and collinearity removal
   are *not* re-run here — inside the fold the only selector is LASSO.
4. **Classifier handling.**  All eleven classifiers are fitted in the same fold
   and share the same fold-selected features.  The ``per_fold_best_mean``
   statistic — the maximum over the eleven models *within each fold*, averaged
   over folds — is **exploratory**: it is optimistically biased, because the
   model is chosen on the very folds that are then reported.  The supplement's
   table note says so.  ``best_algo_AUC`` (one fixed model, chosen by its mean
   over all folds) is the honest comparator, and the paper's K-selection values
   are the exploratory ones.

Inputs come from the command line or from the environment (same convention as
the rest of the repository):

===================================  ==============================================
``--features`` / ``HRPCA_K_FEATURES``  pooled per-habitat feature table, needs an ID column
``--labels``   / ``HRPCA_K_LABELS``    CSV with ``ID,label[,group]``
``--out``      / ``HRPCA_K_OUT``       output directory (default ``results/tables``)
===================================  ==============================================

Usage
-----
    python src/11_habitat_number_selection.py \\
        --features <rad_features.csv> --labels <label-CV-0.csv> \\
        --n_folds 10 --n_seeds 10 --sel lasso --out results/tables

Outputs (in ``--out``)
----------------------
``K_selection_folds_long.csv``  one row per (seed, fold, model, AUC)
``K_selection_folds_wide.csv``  one row per (seed, fold), one column per model
``K_selection_summary.csv``     per-model mean / SD / 95% CI, plus the exploratory
                                per-fold-best mean for every model's column set

Environment note
----------------
Use the interpreter recorded in ``requirements.txt``.  With ``threadpoolctl``
2.2.0 plus a newer OpenMP runtime, sklearn's KNN (which goes through the
``ArgKmin`` reduction) raises ``AttributeError`` inside ``threadpoolctl``.  The
archived run used scikit-learn 1.0.2, where this does not occur; if you hit it,
pass ``--models`` without ``KNN``.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Keep BLAS single-threaded, as in the archived run.  This is set before numpy
# is imported so it actually takes effect, and it makes the fold results
# reproducible across machines.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_MAP = {
    "LR": lambda: LogisticRegression(max_iter=2000),
    "SVM": lambda: SVC(probability=True, random_state=0),
    "KNN": lambda: KNeighborsClassifier(),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=50, max_depth=3, min_samples_split=4, random_state=0),
    "ExtraTrees": lambda: ExtraTreesClassifier(
        n_estimators=80, max_depth=3, min_samples_split=2, random_state=0),
    "XGBoost": lambda: __import__("xgboost").XGBClassifier(
        n_estimators=6, objective="binary:logistic", max_depth=3,
        min_child_weight=.2, use_label_encoder=False, eval_metric="error"),
    "LightGBM": lambda: __import__("lightgbm").LGBMClassifier(
        n_estimators=20, max_depth=4, min_child_weight=0.5),
    "MLP": lambda: MLPClassifier(max_iter=1000, random_state=0),
    "AdaBoost": lambda: AdaBoostClassifier(random_state=0),
    "GradientBoost": lambda: GradientBoostingClassifier(random_state=0),
    "NaiveBayes": lambda: GaussianNB(),
}


def load_data(features: Path, labels: Path) -> tuple:
    """Inner-join the feature table to the label table on ``ID``."""
    rad = pd.read_csv(features)
    lab = pd.read_csv(labels)
    for col in ("ID", "label"):
        if col not in lab.columns:
            raise SystemExit(f"label file lacks column {col!r}: {labels}")
    if "ID" not in rad.columns:
        raise SystemExit(f"feature table lacks an 'ID' column: {features}")
    df = rad.merge(lab[["ID", "label"]], on="ID", how="inner")
    feat_cols = [c for c in rad.columns if c != "ID"]
    X = df[feat_cols].values.astype(float)
    y = df["label"].values.astype(int)
    return X, y


def select_features(Xtr, ytr, Xte, method, budget=None):
    """Fit the selector on the **training fold only**, then apply it to both.

    ``lasso`` is the method used for the K selection; ``mrmr`` and ``none`` are
    kept so the archived comparison can be reproduced.
    """
    n = Xtr.shape[1]
    if method == "none" or n <= 1:
        return Xtr, Xte
    if method == "lasso":
        k = min(5, max(2, np.unique(ytr).size * 3))
        try:
            lasso = LassoCV(cv=k, random_state=0, n_jobs=1).fit(Xtr, ytr)
            sel = np.where(lasso.coef_ != 0)[0]
            if len(sel) == 0:
                sel = np.argsort(-np.abs(lasso.coef_))[:min(10, n)]
        except Exception:
            sel = np.argsort(-np.abs(np.corrcoef(Xtr.T, ytr)[:-1, -1]))[:min(10, n)]
        return Xtr[:, sel], Xte[:, sel]
    if method == "mrmr":
        budget = budget or min(30, max(5, int(n * 0.4)))
        try:
            mi = mutual_info_classif(Xtr, ytr, random_state=0)
            chosen, pool = [], list(range(n))
            while len(chosen) < budget and pool:
                best, best_sc = None, -1e9
                for j in pool:
                    red = 0.0
                    if chosen:
                        red = float(np.mean([abs(np.corrcoef(Xtr[:, j], Xtr[:, c])[0, 1])
                                             for c in chosen]))
                    sc = mi[j] - red
                    if sc > best_sc:
                        best, best_sc = j, sc
                chosen.append(best)
                pool.remove(best)
            sel = np.array(chosen)
        except Exception:
            sel = np.argsort(-np.abs(np.corrcoef(Xtr.T, ytr)[:-1, -1]))[:budget]
        return Xtr[:, sel], Xte[:, sel]
    return Xtr, Xte


_X = None
_y = None


def _init_worker(X, y):
    global _X, _y
    _X, _y = X, y


def _process_task(task, sel, names):
    """One (seed, fold): standardise + select + fit, all on the training fold."""
    seed, fold, tr, te = task
    Xtr, Xte = _X[tr], _X[te]
    ytr, yte = _y[tr], _y[te]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    Xtr, Xte = select_features(Xtr, ytr, Xte, sel)
    auc = {}
    for name in names:
        try:
            est = MODEL_MAP[name]()
            est.fit(Xtr, ytr)
        except Exception:
            continue
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(Xte)
            y_score = proba[:, proba.shape[1] - 1]
        else:
            y_score = est.decision_function(Xte)
        try:
            auc[name] = roc_auc_score(yte, y_score)
        except Exception:
            auc[name] = np.nan
    return seed, fold, auc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=os.environ.get("HRPCA_K_FEATURES"))
    ap.add_argument("--labels", default=os.environ.get("HRPCA_K_LABELS"))
    ap.add_argument("--out", default=os.environ.get("HRPCA_K_OUT", "results/tables"))
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--sel", default="lasso", choices=["lasso", "mrmr", "none"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if not args.features or not args.labels:
        ap.error("--features and --labels are required "
                 "(or set HRPCA_K_FEATURES / HRPCA_K_LABELS)")
    features, labels = Path(args.features), Path(args.labels)
    if not features.exists():
        raise SystemExit(f"feature table not found: {features}")
    if not labels.exists():
        raise SystemExit(f"label file not found: {labels}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_data(features, labels)
    _init_worker(X, y)
    names = list(args.models) if args.models else list(MODEL_MAP)
    names = [n for n in names if n in MODEL_MAP]
    print(f"samples={len(y)}  features={X.shape[1]}  positives={int(y.sum())}")
    print(f"selectors={args.sel}  models={len(names)}  "
          f"folds={args.n_folds}  seeds={args.n_seeds}")

    tasks = []
    for seed in range(args.n_seeds):
        kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(kf.split(X, y)):
            tasks.append((seed, fold, tr, te))

    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(X, y)) as ex:
            gen = ex.map(functools.partial(_process_task, sel=args.sel, names=names),
                         tasks)
            for seed, fold, auc in gen:
                for name, a in auc.items():
                    rows.append([seed, fold, name, a])
    else:
        for task in tasks:
            seed, fold, auc = _process_task(task, args.sel, names)
            for name, a in auc.items():
                rows.append([seed, fold, name, a])

    long = pd.DataFrame(rows, columns=["seed", "fold", "model", "auc"])
    long.to_csv(out / "K_selection_folds_long.csv", index=False, encoding="utf-8-sig")

    wide = long.pivot_table(index=["seed", "fold"], columns="model", values="auc")
    wide = wide.reset_index()
    wide.to_csv(out / "K_selection_folds_wide.csv", index=False, encoding="utf-8-sig")

    stats = []
    for m in sorted(long["model"].unique()):
        v = pd.to_numeric(long[long["model"] == m]["auc"], errors="coerce").dropna()
        if not len(v):
            continue
        stats.append({"Model": m, "n": len(v),
                      "mean": round(float(v.mean()), 4),
                      "std": round(float(v.std(ddof=1)), 4),
                      "min": round(float(v.min()), 4),
                      "max": round(float(v.max()), 4),
                      "CI_low": round(float(np.percentile(v, 2.5)), 4),
                      "CI_high": round(float(np.percentile(v, 97.5)), 4)})
    stat_df = pd.DataFrame(stats).sort_values("mean", ascending=False)

    model_cols = [c for c in wide.columns if c not in ("seed", "fold")]
    per_fold_best = float(wide[model_cols].max(axis=1).mean())
    best_algo = stat_df.iloc[0]
    stat_df.to_csv(out / "K_selection_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== per-model AUC over "
          f"{args.n_seeds * args.n_folds} folds ===")
    print(stat_df.to_string(index=False))
    print(f"\nbest_algo_AUC (one fixed model, honest)      = {best_algo['mean']:.4f} "
          f"({best_algo['Model']})")
    print(f"per_fold_best_mean (EXPLORATORY, optimistic) = {per_fold_best:.4f}")
    print("  -> the paper's K-selection values are the exploratory ones; see the "
          "table note.")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
