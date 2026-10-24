#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-case SHAP decomposition of the combined model — **v3, clinic as one block**.

This is the script that produces the SHAP matrices behind **Fig. 6 / Fig. S10**.

Combined model, on the probability scale (published coefficients):

    L = -4.248788 + 4.062609·p_Clinic + 2.593122·p_H2 + 3.934389·p_H4

    BASE = -4.248788 + 4.062609·0.297739 + 2.593122·0.296332 + 3.934389·0.296558
         = -1.1039 log-odds  (p = 0.2490)

Components
----------
* H4: the 18 features of the H4 model, each ``w_H4·Δp_H4`` split in proportion
  to ``β_k(z_k − E_k) / Σ β(z − E)``
* H2: the 7 features of the H2 model, same rule
* Clinic: **one** block, ``w_CLIN · (p_Clinic − p_Clinic(BASE))``

**26 items in total (18 + 7 + 1).**  Note that the label key in Supplemental
Note S1 and in ``TableS22_Fig6_label_identity.csv`` splits the clinic block into
``ClinISUP`` / ``ClinPIRADS`` (27 items); that is the alternative produced by
``shap_FINAL_v2_alt_clinic_split.py``, which this repository ships side by side.

Closure is asserted: ``BASE + Σ SHAP = L`` to < 1e-9.

Paths
-----
================================  ==========================================
``HRPCA_SHAP_FEATURES``           pooled ``rad_features.csv``
``HRPCA_SHAP_LABELS``             ``label.csv`` (``ID, group, label``)
``HRPCA_SHAP_CLINIC``             ``Clinic.csv``
``HRPCA_SHAP_MODELS``             directory with the two ``*__LR_label.pkl`` lists
``HRPCA_SHAP_RESULTS``            project ``results/`` with ``*_LR_{train,test}.csv``
``HRPCA_SHAP_OUT``                output root; writes ``h2h4_final/``
================================  ==========================================

Outputs (``$HRPCA_SHAP_OUT/h2h4_final/``)
----------------------------------------
``SH_all.npy``  ``(1009, 26)`` float64 SHAP matrix
``SH_feature_names.csv``  the 26 names, in column order
``meta_all.csv``  per-case ``ID, group, label, p2, p4, pc, p_fusion, f_x``
``base.csv``  the log-odds base value

Usage
-----
    python src/shap/shap_FINAL_v3.py
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- paths: set these in the environment (see src/shap/README.md) -----------
P_FEATURES = os.environ.get("HRPCA_SHAP_FEATURES", "")
P_LABELS = os.environ.get("HRPCA_SHAP_LABELS", "")
P_CLINIC = os.environ.get("HRPCA_SHAP_CLINIC", "")
P_MODELS = os.environ.get("HRPCA_SHAP_MODELS", "")
P_RESULTS = os.environ.get("HRPCA_SHAP_RESULTS", "")
P_OUT = os.environ.get("HRPCA_SHAP_OUT", "")


def need(value: str, var: str) -> str:
    if not value:
        raise SystemExit(f"{var} is not set — see src/shap/README.md")
    return value


FEATURES = need(P_FEATURES, "HRPCA_SHAP_FEATURES")
LABELS = need(P_LABELS, "HRPCA_SHAP_LABELS")
CLINIC = need(P_CLINIC, "HRPCA_SHAP_CLINIC")
MODELS = need(P_MODELS, "HRPCA_SHAP_MODELS")
RES = need(P_RESULTS, "HRPCA_SHAP_RESULTS")
OUT = os.path.join(need(P_OUT, "HRPCA_SHAP_OUT"), "h2h4_final")
os.makedirs(OUT, exist_ok=True)

W0, W_CLIN, W_H2, W_H4 = -4.248788185663511, 4.062608661920751, 2.593122775462074, 3.9343891733424328
PB_CLIN, PB_H2, PB_H4 = 0.297739, 0.296332, 0.296558
BASE = W0 + W_CLIN * PB_CLIN + W_H2 * PB_H2 + W_H4 * PB_H4
print(f"BASE = {BASE:.4f}")

lab = pd.read_csv(LABELS, dtype=str)
lab["ID"] = lab["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
lab["label"] = lab["label"].astype(int)
fH2 = [str(x) for x in pickle.load(open(os.path.join(MODELS, "HabitatH2__LR_label.pkl"), "rb"))]
fH4 = [str(x) for x in pickle.load(open(os.path.join(MODELS, "HabitatH4__LR_label.pkl"), "rb"))]
RF = pd.read_csv(FEATURES, usecols=["ID"] + sorted(set(fH2 + fH4)))
RF["ID"] = RF["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()

# Clinic enters as a single block, built from the two variables the project's
# Clinic model actually uses (PI-RADS + ISUP).
clin = pd.read_csv(CLINIC, dtype=str)
cid = [c for c in clin.columns if c.strip().upper() in ("ID", "CASE", "PATIENT")][0]
clin["ID"] = clin[cid].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
CLIN_VARS = ["PI-RADS", "ISUP"]


def rd(fn, tag):
    d = pd.read_csv(os.path.join(RES, fn), dtype=str)
    d["ID"] = d.iloc[:, 0].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
    d[tag] = pd.to_numeric(d["label-1"], errors="coerce")
    return d[["ID", tag]]


proj = pd.concat([rd("HabitatH2_LR_train.csv", "p2"), rd("HabitatH2_LR_test.csv", "p2")]).drop_duplicates("ID")
proj = proj.merge(pd.concat([rd("HabitatH4_LR_train.csv", "p4"), rd("HabitatH4_LR_test.csv", "p4")]).drop_duplicates("ID"), on="ID")
proj = proj.merge(pd.concat([rd("Clinic_LR_train.csv", "pc"), rd("Clinic_LR_test.csv", "pc")]).drop_duplicates("ID"), on="ID")

m = lab.merge(RF, on="ID").merge(clin[["ID"] + CLIN_VARS], on="ID").merge(proj, on="ID")
m = m.dropna(subset=CLIN_VARS + fH2 + fH4 + ["p2", "p4", "pc"]).reset_index(drop=True)
trm = m.group.str.lower().str.contains("train").values
print(f"cases {len(m)} | train {int(trm.sum())}")
L = W0 + W_CLIN * m.pc.values + W_H2 * m.p2.values + W_H4 * m.p4.values


def fit(feats):
    X = m[feats].values.astype(float)
    sc = StandardScaler().fit(X[trm])
    lr = LogisticRegression(max_iter=5000, random_state=42).fit(sc.transform(X[trm]),
                                                               m.label.values[trm])
    Z = sc.transform(X)
    E = Z[trm].mean(axis=0)
    return dict(b=lr.coef_.ravel(), Z=Z, E=E)


def alloc(s, w, p_obs, p_base):
    zE = (s["Z"] - s["E"]) @ s["b"]
    d = w * (p_obs - p_base)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(zE)[:, None] > 1e-12,
                         (s["Z"] - s["E"]) * s["b"] / zE[:, None], 0.0)
    return np.nan_to_num(d[:, None] * ratio)


SH_H4 = alloc(fit(fH4), W_H4, m.p4.values, PB_H4)
SH_H2 = alloc(fit(fH2), W_H2, m.p2.values, PB_H2)
# Clinic as ONE block: SHAP = w_CLIN · (p_Clinic − p_Clinic(base))
SH_C = (W_CLIN * (m.pc.values - PB_CLIN)).reshape(-1, 1)

SH = np.column_stack([SH_H4, SH_H2, SH_C])
names = fH4 + fH2 + ["Clinic probability"]
resid = np.abs(BASE + SH.sum(axis=1) - L)
print(f"[closure] max|BASE + SHAP - L| = {resid.max():.3e}")
assert resid.max() < 1e-9
print("closure OK | items:", SH.shape[1], "(H4 18 + H2 7 + Clinic 1)")

IDS = ["case_a", "case_b", "case_c", "case_d", "case_e"]
t = m[["ID", "group", "label", "p2", "p4", "pc"]].copy()
t["p_fusion"] = 1 / (1 + np.exp(-L))
t["f_x"] = L
print("\n=== the five cases ===")
pd.set_option("display.width", 200)
print(t[t.ID.isin(IDS)].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
tgt = {"case_a": 0.7889, "case_b": 0.0358, "case_c": 0.6144,
       "case_d": 0.0510, "case_e": 0.0508}
print("\nagainst the paper:")
for _, r in t[t.ID.isin(IDS)].iterrows():
    print(f"  {r.ID}: {r.p_fusion:.4f} vs {tgt[r.ID]:.4f} -> d {r.p_fusion - tgt[r.ID]:+.4f}")

np.save(os.path.join(OUT, "SH_all.npy"), SH)
pd.Series(names).to_csv(os.path.join(OUT, "SH_feature_names.csv"), index=False, header=["feature"])
t.to_csv(os.path.join(OUT, "meta_all.csv"), index=False, encoding="utf-8-sig")
pd.Series({"base": BASE}).to_csv(os.path.join(OUT, "base.csv"), index=False)
print("\noutput:", OUT)
