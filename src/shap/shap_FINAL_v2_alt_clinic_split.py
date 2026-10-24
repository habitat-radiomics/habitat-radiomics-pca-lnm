#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-case SHAP decomposition — **v2, clinic split into ISUP and PI-RADS**.

This is the alternative form: instead of the clinic probability entering as one
block, it is expanded into its two standardised inputs.  Same combined model and
same base value as ``shap_FINAL_v3.py``:

    BASE = -4.248788 + 4.062609·0.297739 + 2.593122·0.296332 + 3.934389·0.296558
         = -1.1039 log-odds

* H4: 18 features  · H2: 7 features  · Clinic: **ISUP + PI-RADS (2 items)**
* **27 items in total.**  This is the form assumed by the label key in
  Supplemental Note S1 (``ClinISUP`` / ``ClinPIRADS``) and by
  ``TableS22_Fig6_label_identity.csv`` ("pooled contribution of the remaining
  19 features").

Closure is asserted: ``BASE + Σ SHAP = L`` to < 1e-9.

Writes to its own directory (``h2h4_final_v2_clinic_split/``) so that running it
does not overwrite the 26-item output of ``shap_FINAL_v3.py``.

Paths
-----
================================  ==========================================
``HRPCA_SHAP_FEATURES``           pooled ``rad_features.csv``
``HRPCA_SHAP_LABELS``             ``label.csv`` (``ID, group, label``)
``HRPCA_SHAP_CLINIC``             ``Clinic.csv``
``HRPCA_SHAP_MODELS``             directory with the two ``*__LR_label.pkl`` lists
``HRPCA_SHAP_RESULTS``            project ``results/`` with ``*_LR_{train,test}.csv``
``HRPCA_SHAP_OUT``                output root; writes ``h2h4_final_v2_clinic_split/``
================================  ==========================================

Usage
-----
    python src/shap/shap_FINAL_v2_alt_clinic_split.py
"""
import os
import pickle
import re
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
OUT = os.path.join(need(P_OUT, "HRPCA_SHAP_OUT"), "h2h4_final_v2_clinic_split")
os.makedirs(OUT, exist_ok=True)

W0, W_CLIN, W_H2, W_H4 = -4.248788185663511, 4.062608661920751, 2.593122775462074, 3.9343891733424328
# component probabilities at the published base value
PB_CLIN, PB_H2, PB_H4 = 0.297739, 0.296332, 0.296558
BASE = W0 + W_CLIN * PB_CLIN + W_H2 * PB_H2 + W_H4 * PB_H4
print(f"BASE = {BASE:.4f}  (p = {1/(1+np.exp(-BASE)):.4f})")
assert abs(BASE - (-1.1039)) < 5e-4, f"BASE mismatch: {BASE}"

lab = pd.read_csv(LABELS, dtype=str)
lab["ID"] = lab["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
lab["label"] = lab["label"].astype(int)
fH2 = [str(x) for x in pickle.load(open(os.path.join(MODELS, "HabitatH2__LR_label.pkl"), "rb"))]
fH4 = [str(x) for x in pickle.load(open(os.path.join(MODELS, "HabitatH4__LR_label.pkl"), "rb"))]
RF = pd.read_csv(FEATURES, usecols=["ID"] + sorted(set(fH2 + fH4)))
RF["ID"] = RF["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
clin = pd.read_csv(CLINIC, dtype=str)
cid = [c for c in clin.columns if c.strip().upper() in ("ID", "CASE", "PATIENT")][0]
clin["ID"] = clin[cid].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
cand = sorted([c for c in clin.columns if re.search(r"isup|pirads|pi-rads|pi_rads", c, re.I)],
              key=lambda c: (0 if re.search(r"isup", c, re.I) else 1))


def rd(fn, tag):
    d = pd.read_csv(os.path.join(RES, fn), dtype=str)
    d["ID"] = d.iloc[:, 0].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
    d[tag] = pd.to_numeric(d["label-1"], errors="coerce")
    return d[["ID", tag]]


proj = pd.concat([rd("HabitatH2_LR_train.csv", "p2"), rd("HabitatH2_LR_test.csv", "p2")]).drop_duplicates("ID")
proj = proj.merge(pd.concat([rd("HabitatH4_LR_train.csv", "p4"), rd("HabitatH4_LR_test.csv", "p4")]).drop_duplicates("ID"), on="ID")
proj = proj.merge(pd.concat([rd("Clinic_LR_train.csv", "pc"), rd("Clinic_LR_test.csv", "pc")]).drop_duplicates("ID"), on="ID")

m = lab.merge(RF, on="ID").merge(clin[["ID"] + cand], on="ID").merge(proj, on="ID")
m = m.dropna(subset=cand + fH2 + fH4 + ["p2", "p4", "pc"]).reset_index(drop=True)
trm = m.group.str.lower().str.contains("train").values
L = W0 + W_CLIN * m.pc.values + W_H2 * m.p2.values + W_H4 * m.p4.values


def fit(feats):
    X = m[feats].values.astype(float)
    sc = StandardScaler().fit(X[trm])
    lr = LogisticRegression(max_iter=5000, random_state=42).fit(sc.transform(X[trm]),
                                                               m.label.values[trm])
    Z = sc.transform(X)
    E = Z[trm].mean(axis=0)
    return dict(b=lr.coef_.ravel(), Z=Z, E=E)


sH2, sH4 = fit(fH2), fit(fH4)
Xc = m[cand].values.astype(float)
scc = StandardScaler().fit(Xc[trm])
lrc = LogisticRegression(max_iter=5000, random_state=42).fit(scc.transform(Xc[trm]),
                                                             m.label.values[trm])
Zc = scc.transform(Xc)
Ec = Zc[trm].mean(axis=0)
bc = lrc.coef_.ravel()


def alloc(s, w, p_obs, p_base):
    zE = (s["Z"] - s["E"]) @ s["b"]
    d = w * (p_obs - p_base)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(zE)[:, None] > 1e-12,
                         (s["Z"] - s["E"]) * s["b"] / zE[:, None], 0.0)
    return np.nan_to_num(d[:, None] * ratio)


SH_H4 = alloc(sH4, W_H4, m.p4.values, PB_H4)
SH_H2 = alloc(sH2, W_H2, m.p2.values, PB_H2)
zEc = (Zc - Ec) @ bc
d_C = W_CLIN * (m.pc.values - PB_CLIN)
with np.errstate(divide="ignore", invalid="ignore"):
    rC = np.where(np.abs(zEc)[:, None] > 1e-12, (Zc - Ec) * bc / zEc[:, None], 0.0)
SH_C = np.nan_to_num(d_C[:, None] * rC)

SH = np.column_stack([SH_H4, SH_H2, SH_C])
names = fH4 + fH2 + [f"Clinical: {c}" for c in cand]
resid = np.abs(BASE + SH.sum(axis=1) - L)
print(f"[closure] max|BASE + SHAP - L| = {resid.max():.3e}")
assert resid.max() < 1e-9, "decomposition does not close"
print(f"closure OK | items: {SH.shape[1]} (H4 18 + H2 7 + Clinical {len(cand)})")

IDS = ["case_a", "case_b", "case_c", "case_d", "case_e"]
t = m[["ID", "group", "label", "p2", "p4", "pc"]].copy()
t["p_fusion"] = 1 / (1 + np.exp(-L))
t["f_x"] = L
s = t[t.ID.isin(IDS)]
print("\n=== the five cases ===")
pd.set_option("display.width", 200)
print(s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
tgt = {"case_a": (1.3182, 0.7889), "case_b": (-3.2921, 0.0358), "case_c": (0.4658, 0.6144),
       "case_d": (-2.9228, 0.0510), "case_e": (-2.9271, 0.0508)}
print("\nagainst the targets:")
ok = True
for _, r in s.iterrows():
    fl, fp = tgt[r.ID]
    d1, d2 = r.f_x - fl, r.p_fusion - fp
    flag = "OK" if abs(d1) < 0.01 and abs(d2) < 0.005 else "MISMATCH"
    if flag != "OK":
        ok = False
    print(f"  {r.ID}: f(x) {r.f_x:+.4f} (target {fl:+.4f}, d {d1:+.4f}) | "
          f"p {r.p_fusion:.4f} (target {fp:.4f}, d {d2:+.4f}) {flag}")
print("\nall passed" if ok else "\nsome still differ")

np.save(os.path.join(OUT, "SH_all.npy"), SH)
pd.Series(names).to_csv(os.path.join(OUT, "SH_feature_names.csv"), index=False, header=["feature"])
t.to_csv(os.path.join(OUT, "meta_all.csv"), index=False, encoding="utf-8-sig")
pd.Series({"base": BASE}).to_csv(os.path.join(OUT, "base.csv"), index=False)
print("BASE written to:", os.path.join(OUT, "base.csv"), "| output:", OUT)
