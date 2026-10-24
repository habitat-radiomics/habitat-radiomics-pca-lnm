#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which clinical variables does the project's Clinic model actually use?

Enumerates every non-empty subset of the clinical columns (127 combinations for
seven variables), refits a standardised logistic regression on the training
cohort for each subset, and compares the resulting probabilities with the
project's own ``results/Clinic_LR_{train,test}.csv`` output.

The answer for the archived run: **PI-RADS + ISUP** (r = 0.9998, mean absolute
difference 0.0030).  This is the check that justifies treating the clinic term as
those two variables in ``shap_FINAL_v2_alt_clinic_split.py``.

Paths
-----
==========================  ============================================
``HRPCA_SHAP_LABELS``       ``label.csv`` (``ID, group, label``)
``HRPCA_SHAP_CLINIC``       ``Clinic.csv``
``HRPCA_SHAP_RESULTS``      project ``results/`` with ``Clinic_LR_*.csv``
==========================  ============================================

Usage
-----
    python src/shap/find_clinic_model_vars.py
"""
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- paths: set these in the environment (see src/shap/README.md) -----------
P_LABELS = os.environ.get("HRPCA_SHAP_LABELS", "")
P_CLINIC = os.environ.get("HRPCA_SHAP_CLINIC", "")
P_RESULTS = os.environ.get("HRPCA_SHAP_RESULTS", "")


def need(value: str, var: str) -> str:
    if not value:
        raise SystemExit(f"{var} is not set — see src/shap/README.md")
    return value


LABELS = need(P_LABELS, "HRPCA_SHAP_LABELS")
CLINIC = need(P_CLINIC, "HRPCA_SHAP_CLINIC")
RES = need(P_RESULTS, "HRPCA_SHAP_RESULTS")

lab = pd.read_csv(LABELS, dtype=str)
lab["ID"] = lab["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
lab["label"] = lab["label"].astype(int)
clin = pd.read_csv(CLINIC, dtype=str)
cid = [c for c in clin.columns if c.strip().upper() in ("ID", "CASE", "PATIENT")][0]
clin["ID"] = clin[cid].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
VARS = [c for c in clin.columns if c != "ID"]
print("clinical variables:", VARS)


def rd(fn, tag):
    d = pd.read_csv(os.path.join(RES, fn), dtype=str)
    d["ID"] = d.iloc[:, 0].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
    d[tag] = pd.to_numeric(d["label-1"], errors="coerce")
    return d[["ID", tag]]


proj = pd.concat([rd("Clinic_LR_train.csv", "pc"),
                  rd("Clinic_LR_test.csv", "pc")]).drop_duplicates("ID")
m = lab.merge(clin, on="ID").merge(proj, on="ID")
print(f"cases {len(m)} | with a project Clinic prediction {m.pc.notna().sum()}")

best = []
for r in range(1, len(VARS) + 1):
    for combo in combinations(VARS, r):
        cols = list(combo)
        d = m[cols + ["pc", "label", "group"]].apply(pd.to_numeric, errors="coerce")
        d["group"] = m.group.values
        d["pc"] = m.pc.values
        d = d.dropna()
        if len(d) < 300:
            continue
        tr = d.group.str.lower().str.contains("train").values
        if tr.sum() < 100 or d.pc.isna().all():
            continue
        sc = StandardScaler().fit(d[cols].values[tr])
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(d[cols].values[tr]),
                                                   d.label.values[tr])
        p = lr.predict_proba(sc.transform(d[cols].values))[:, 1]
        cc = np.corrcoef(p, d.pc.values)[0, 1]
        mad = float(np.abs(p - d.pc.values).mean())
        best.append((cc, mad, combo, len(d), int(tr.sum()), lr))

best.sort(key=lambda x: (-x[0], x[1]))
print("\n=== best-matching clinical variable subsets (top 10) ===")
for cc, mad, combo, n, ntr, lr in best[:10]:
    print(f"  r {cc:.6f} | mean abs diff {mad:.5f} | n={n} train={ntr} | {' + '.join(combo)}")
cc, mad, combo, n, ntr, lr = best[0]
print(f"\nbest subset: {' + '.join(combo)}")
print("  standardised coefficients:", dict(zip(combo, np.round(lr.coef_.ravel(), 4))))
print("  intercept:", round(float(lr.intercept_[0]), 4))
