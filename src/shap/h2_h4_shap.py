#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2 / H4 habitat models — per-model SHAP summary (beeswarm + mean|SHAP|).

Reproduces the two panels of **Fig. 5** and the tables ``SHAP_H2_ranking.csv`` /
``SHAP_H4_ranking.csv``.

Data flow
---------
* feature names  <- ``HabitatH2__LR_label.pkl`` / ``HabitatH4__LR_label.pkl``
  (7 and 18 names; the pkl stores the **name list**, not a fitted model)
* feature values  <- the pooled ``rad_features.csv``
* labels          <- ``label.csv`` (``ID, group, label``); only ``group == train``
  is used, n = 398
* model           <- StandardScaler + LogisticRegression refitted on the training
  cohort with these feature names (the project's CV selected LR for both)

Paths
-----
===============  ====================================================
``HRPCA_SHAP_FEATURES``  pooled ``rad_features.csv``
``HRPCA_SHAP_LABELS``    ``label.csv`` (``ID, group, label``)
``HRPCA_SHAP_MODELS``    directory holding the two ``*__LR_label.pkl`` name lists
``HRPCA_SHAP_OUT``       output root; this script writes ``h2_h4_shap/`` inside it
===============  ====================================================

Usage
-----
    python src/shap/h2_h4_shap.py
"""
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- paths: set these in the environment (see src/shap/README.md) -----------
P_FEATURES = os.environ.get("HRPCA_SHAP_FEATURES", "")
P_LABELS = os.environ.get("HRPCA_SHAP_LABELS", "")
P_MODELS = os.environ.get("HRPCA_SHAP_MODELS", "")
P_OUT = os.environ.get("HRPCA_SHAP_OUT", "")


def need(value: str, var: str) -> str:
    if not value:
        raise SystemExit(f"{var} is not set — see src/shap/README.md")
    return value


LAB = need(P_LABELS, "HRPCA_SHAP_LABELS")
P = need(P_MODELS, "HRPCA_SHAP_MODELS")
OUT = os.path.join(need(P_OUT, "HRPCA_SHAP_OUT"), "h2_h4_shap")
FEATURES = need(P_FEATURES, "HRPCA_SHAP_FEATURES")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": ["Arial", "Microsoft YaHei"],
                     "font.size": 9, "figure.dpi": 300, "savefig.dpi": 300})

lab = pd.read_csv(LAB, dtype=str)
lab["ID"] = lab["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True)
lab["label"] = lab["label"].astype(int)


def pretty(f):
    h = re.search(r"_h(\d+)$", f).group(1)
    b = f[: f.rfind("_h" + h)]
    b = re.sub(r"^log_sigma_(\d)_(\d)_mm_3D_", "LoG \u03c3\\1 \u00b7 ", b)
    b = re.sub(r"^wavelet_([A-Z]{3})_", "W-\\1 \u00b7 ", b)
    b = b.replace("original_", "Orig \u00b7 ")
    b = b.replace("glcm_", "GLCM ").replace("glszm_", "GLSZM ").replace("glrlm_", "GLRLM ")
    b = b.replace("gldm_", "GLDM ").replace("firstorder_", "1st ").replace("shape_", "Shape ")
    b = b.replace("ngtdm_", "NGTDM ")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", b).replace("_h" + h, "")


def run(H):
    print(f"\n{'='*70}\n### {H}\n{'='*70}")
    with open(os.path.join(P, f"Habitat{H}__LR_label.pkl"), "rb") as fh:
        feats = [str(x) for x in pickle.load(fh)]
    print(f"features: {len(feats)}")
    rf = pd.read_csv(FEATURES, usecols=["ID"] + feats)
    rf["ID"] = rf["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True)
    m = lab.merge(rf, on="ID", how="inner")
    tr = m[m.group.astype(str).str.lower().str.contains("train")].dropna(subset=feats)
    print(f"training cohort: {len(tr)} cases | pN1: {int(tr.label.sum())}")

    X = tr[feats].values.astype(float)
    y = tr["label"].values
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    lr = LogisticRegression(max_iter=2000, random_state=42).fit(Xs, y)
    auc = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])
    print(f"LR training AUC: {auc:.3f} | coef range {lr.coef_.min():+.2f} ~ {lr.coef_.max():+.2f}")

    expl = shap.LinearExplainer(lr, Xs)
    sv = expl.shap_values(Xs)
    sv = np.asarray(sv)
    base = float(np.asarray(expl.expected_value).ravel()[0])
    print(f"SHAP matrix {sv.shape} | base(log-odds) = {base:+.3f}")

    names = [pretty(f) for f in feats]
    ma = np.abs(sv).mean(axis=0)
    rk = pd.DataFrame({"feature": feats, "short": names, "mean_abs_SHAP": ma}).sort_values(
        "mean_abs_SHAP", ascending=False).reset_index(drop=True)
    rk.to_csv(os.path.join(OUT, f"SHAP_{H}_ranking.csv"), index=False, encoding="utf-8-sig")
    det = pd.DataFrame(sv, columns=[f"SHAP_{f}" for f in feats])
    det.insert(0, "ID", tr["ID"].values)
    det.insert(1, "label", y)
    det.to_csv(os.path.join(OUT, f"SHAP_{H}_values_train.csv"), index=False, encoding="utf-8-sig")
    print(rk.head(8).to_string(index=False))

    # — panel 1: beeswarm (stand-alone, PNG + SVG); widened x-axis with padding —
    _pw, _ph = 13.5, 0.42 * len(feats) + 1.9
    plt.figure()
    shap.summary_plot(sv, Xs, feature_names=names, show=False, max_display=len(feats),
                      plot_size=(_pw, _ph))          # plot_size, not figsize
    _ax = plt.gca()
    _lo, _hi = _ax.get_xlim()
    _pad = (_hi - _lo) * 0.08
    _ax.set_xlim(_lo - _pad, _hi + _pad)
    plt.title(f"{H} habitat model — SHAP summary (beeswarm)", fontsize=11.5,
              fontweight="bold", loc="left", pad=12)
    plt.tight_layout()
    for ext in [".png", ".svg"]:
        plt.savefig(os.path.join(OUT, f"Fig_SHAP_{H}_beeswarm{ext}"), bbox_inches="tight")
    plt.close("all")

    # — panel 2: mean|SHAP| ranking (stand-alone, PNG + SVG) —
    plt.figure(figsize=(7.8, 0.34 * len(feats) + 1.7))
    order = rk[::-1]
    plt.barh(order.short, order.mean_abs_SHAP, color="#3D8FD1", height=0.62)
    plt.xlabel("mean(|SHAP value|)  [log-odds]", fontsize=10)
    plt.title(f"{H} habitat model — Mean |SHAP| ranking", fontsize=11.5,
              fontweight="bold", loc="left", pad=12)
    plt.tick_params(axis="y", labelsize=9)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    for ext in [".png", ".svg"]:
        plt.savefig(os.path.join(OUT, f"Fig_SHAP_{H}_bar{ext}"), bbox_inches="tight")
    plt.close("all")
    print("figures:", os.path.join(OUT, f"Fig_SHAP_{H}_beeswarm.png"), "+",
          f"Fig_SHAP_{H}_bar.png", "(each with .svg)")
    return dict(habitat=H, n_features=len(feats), n_train=len(tr), n_pN1=int(y.sum()),
                train_auc=round(auc, 4), base_logodds=round(base, 4),
                top=rk.feature.iloc[0], top_mean_abs=round(rk.mean_abs_SHAP.iloc[0], 4))


summary = [run("H2"), run("H4")]
pd.DataFrame(summary).to_csv(os.path.join(OUT, "summary_H2_H4.csv"), index=False,
                             encoding="utf-8-sig")
print("\n=== summary ===")
print(pd.DataFrame(summary).to_string(index=False))
print("\noutput:", OUT)
