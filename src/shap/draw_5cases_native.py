#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-case force plots for the five illustrative cases (native ``shap`` rendering).

Reads the matrices written by ``shap_FINAL_v3.py`` and draws one force plot per
case with ``shap.force_plot(..., matplotlib=True)``:

* the top 8 features by |SHAP| plus an ``N others`` segment that **does**
  participate in f(x)
* labels of the form ``(1)LoG...h4=+4.49`` — index + short name + ``...`` +
  habitat + ``=`` + the feature value (z-score)
* ``higher <-`` / ``-> lower`` placed at both ends of the strip, ``f(x)`` and
  ``base value`` lifted clear of the axis
* 15 x 3.4 in at 300 dpi, exported as PNG + SVG

The item count follows whatever ``SH_all.npy`` contains, so this script works
with both the 26-item (v3) and the 27-item (v2) decompositions.

Paths
-----
==============================  ============================================
``HRPCA_SHAP_OUT``              output root; reads ``h2h4_final/`` and writes ``native/``
``HRPCA_SHAP_FEATURES``         pooled ``rad_features.csv``
``HRPCA_SHAP_CLINIC``           ``Clinic.csv``
``HRPCA_SHAP_MODELS``           directory with the two ``*__LR_label.pkl`` lists
``HRPCA_SHAP_EXPORT``           optional: extra directory to copy the PNG/SVG into
==============================  ============================================

Usage
-----
    python src/shap/draw_5cases_native.py
"""
import os
import pickle
import re
import shutil

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler

# --- paths: set these in the environment (see src/shap/README.md) -----------
P_FEATURES = os.environ.get("HRPCA_SHAP_FEATURES", "")
P_CLINIC = os.environ.get("HRPCA_SHAP_CLINIC", "")
P_MODELS = os.environ.get("HRPCA_SHAP_MODELS", "")
P_OUT = os.environ.get("HRPCA_SHAP_OUT", "")
P_EXPORT = os.environ.get("HRPCA_SHAP_EXPORT", "")


def need(value: str, var: str) -> str:
    if not value:
        raise SystemExit(f"{var} is not set — see src/shap/README.md")
    return value


plt.rcParams.update({"font.family": ["Arial", "Microsoft YaHei"],
                     "font.size": 9, "figure.dpi": 300, "savefig.dpi": 300})

FEATURES = need(P_FEATURES, "HRPCA_SHAP_FEATURES")
CLINIC = need(P_CLINIC, "HRPCA_SHAP_CLINIC")
MODELS = need(P_MODELS, "HRPCA_SHAP_MODELS")
ROOT = need(P_OUT, "HRPCA_SHAP_OUT")
SRC = os.path.join(ROOT, "h2h4_final")
LOCAL = os.path.join(ROOT, "native")
os.makedirs(LOCAL, exist_ok=True)
CIRC = "①②③④⑤⑥⑦⑧⑨⑩"
NDISP = 8

SH = np.load(os.path.join(SRC, "SH_all.npy"))
names = pd.read_csv(os.path.join(SRC, "SH_feature_names.csv"))["feature"].tolist()
meta = pd.read_csv(os.path.join(SRC, "meta_all.csv"), dtype={"ID": str})
base = float(pd.read_csv(os.path.join(SRC, "base.csv")).iloc[0, 0])

# standardised feature values, on the same scale as the SHAP values
fH2 = [str(x) for x in pickle.load(open(os.path.join(MODELS, "HabitatH2__LR_label.pkl"), "rb"))]
fH4 = [str(x) for x in pickle.load(open(os.path.join(MODELS, "HabitatH4__LR_label.pkl"), "rb"))]
dH2 = pd.read_csv(FEATURES, usecols=["ID"] + fH2)
dH4 = pd.read_csv(FEATURES, usecols=["ID"] + fH4)
for d in (dH2, dH4):
    d["ID"] = d["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
clin = pd.read_csv(CLINIC, dtype=str)
cid = [c for c in clin.columns if c.strip().upper() in ("ID", "CASE", "PATIENT")][0]
clin["ID"] = clin[cid].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).str.strip()
trm = meta.group.str.lower().str.contains("train").values

Z = {}
for feats, d in [(fH4, dH4), (fH2, dH2)]:
    mm = meta[["ID"]].merge(d, on="ID", how="left")
    X = mm[feats].values.astype(float)
    sc = StandardScaler().fit(X[trm])
    Z.update({f: sc.transform(X)[:, i] for i, f in enumerate(feats)})
# the clinic block is the project's Clinic_LR output probability
Z["Clinic probability"] = meta["pc"].astype(float).values
# the v2 form carries the two clinical inputs instead of the single block
for c in ("ISUP", "PI-RADS"):
    if f"Clinical: {c}" in names:
        Z[f"Clinical: {c}"] = pd.to_numeric(
            meta[["ID"]].merge(clin, on="ID", how="left")[c], errors="coerce").values
Zm = np.column_stack([Z[f] for f in names])
print("z matrix:", Zm.shape, "| items:", len(names))

CLS = {"glcm": "GLCM", "glszm": "GLSZM", "glrlm": "GLRLM", "gldm": "GLDM",
       "ngtdm": "NGTDM", "firstorder": "1st-order", "shape": "Shape"}


def label_parts(f):
    """Split into (filter/transform, full feature name, habitat) for the 3-line label."""
    if f == "Clinic probability":
        return "Clinic", "PI-RADS + ISUP", ""
    if f.startswith("Clinical:"):
        return "Clinical", f.split(":")[1].strip(), ""
    h = re.search(r"_h(\d+)$", f)
    hh = f"H{h.group(1)}" if h else ""
    b = f[: f.rfind("_h" + h.group(1))] if h else f
    pre, rest = "", b
    m1 = re.match(r"^log_sigma_(\d)_(\d)_mm_3D_(.*)$", b)
    m2 = re.match(r"^wavelet_([A-Z]{3})_(.*)$", b)
    m3 = re.match(r"^original_(.*)$", b)
    if m1:
        pre, rest = f"Lo G \u03c3{m1.group(1)}", m1.group(3)
    elif m2:
        pre, rest = f"W-{m2.group(1)}", m2.group(2)
    elif m3:
        pre, rest = "Orig", m3.group(1)
    for k, v in CLS.items():
        if rest.startswith(k + "_"):
            rest = v + " " + rest[len(k) + 1:]
            break
    rest = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", rest)
    return pre, rest.strip(), hh


def short_name(f):
    """Short label: first segment + habitat (e.g. WLLL...h3 / LoG...h4 / ClinISUP)."""
    if f == "Clinic probability":
        return "Clinic"
    if f.startswith("Clinical:"):
        return "Clin" + f.split(":")[1].strip().replace("-", "")
    h = re.search(r"_h(\d+)$", f).group(1)
    b = f[: f.rfind("_h" + h)]
    b = re.sub(r"^log_sigma_\d_\d_mm_3D_", "LoG_", b)
    b = re.sub(r"^wavelet_([A-Z]{3})_", r"W\1_", b)
    b = b.replace("original_", "")
    return f"{b.split('_')[0]}\u2026h{h}"


shorts = [short_name(f) for f in names]

CASES = [("a", "case_a"), ("b", "case_b"), ("c", "case_c"),
         ("d", "case_d"), ("e", "case_e")]
made = []
for tag, pid in CASES:
    r = meta[meta.ID == pid]
    if not len(r):
        continue
    i = r.index[0]
    sh = SH[i]
    xz = Zm[i]
    o = np.argsort(-np.abs(sh))
    top = list(o[:NDISP])
    rest = [ix for ix in range(len(sh)) if ix not in top]
    sv8 = np.concatenate([sh[top], [sh[rest].sum() if rest else 0.0]])
    x8 = np.concatenate([xz[top], [0.0]])   # 0 for others (NaN would drop the segment)
    nm8 = [shorts[j] for j in top] + [f"{len(rest)} others"]

    fig = plt.figure()
    shap.force_plot(base, sv8, x8, feature_names=nm8, matplotlib=True, show=False,
                    figsize=(15, 3.4), plot_cmap="RdBu")
    ax = plt.gca()
    # labels: short form, matched on the (unique) feature value
    for t in ax.texts:
        old = t.get_text()
        if " = " not in old:
            continue
        nmo = old.split(" = ")[0].strip()
        if nmo.endswith("others"):
            t.set_text(f"{len(names)-NDISP} others")
            continue
        try:
            val = float(old.split(" = ")[1])
        except Exception:
            continue
        x8f = np.where(np.isnan(x8), 1e6, x8)          # NaN (others) excluded
        k = int(np.argmin(np.abs(x8f - val)))
        fi = top[k] if k < len(top) else None
        if fi is None:
            continue
        t.set_text(f"{CIRC[k]}{nmo}={val:+.2f}")
    for t in ax.texts:
        s_ = t.get_text().strip().lower()
        if s_ in ("higher", "lower", "$\\leftarrow$", "$\\rightarrow$", "\u2190", "\u2192"):
            t.set_visible(False)
    # lift f(x) / base value clear of the axis and ticks
    for t in ax.texts:
        s_ = t.get_text().strip()
        if s_.lower() in ("f(x)", "base value") or re.fullmatch(r"-?\d+\.\d+", s_):
            x0, y0 = t.get_position()
            t.set_position((x0, float(y0) + 0.05))
    # higher/lower -> both ends of the strip, same height as the band
    bands = []
    for pp in ax.patches:
        if type(pp).__name__ != "Polygon":
            continue
        fc = pp.get_facecolor()
        if len(fc) < 4 or fc[3] < 0.9:
            continue
        bb = pp.get_extents()
        if bb.width > 3:
            bands.append(bb)
    if bands:
        b_red = min(bands, key=lambda b: b.x0 + b.width / 2)
        y_bar = ax.transData.inverted().transform((0, b_red.y0 + b_red.height / 2))[1]
        xa, xb = ax.get_xlim()
        ax.text(xa, y_bar, "higher \u2190", color="#E8174B", fontsize=10, fontweight="bold",
                ha="left", va="center", zorder=8)
        ax.text(xb, y_bar, "\u2192 lower", color="#2E7BCF", fontsize=10, fontweight="bold",
                ha="right", va="center", zorder=8)
    p = 1 / (1 + np.exp(-(base + float(sh.sum()))))
    row = r.iloc[0]
    ax.set_title(f"({tag})  {pid}   {row['group']}   {'pN1' if int(row['label'])==1 else 'pN0'}   "
                 f"p(LNM) = {p:.4f}", fontsize=10.5, fontweight="bold", loc="left", pad=18)
    plt.tight_layout()
    fp = os.path.join(LOCAL, f"Fig_force_fusion_{tag}_{pid}.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.savefig(fp.replace(".png", ".svg"), bbox_inches="tight")   # vector copy
    plt.close("all")
    made.append((tag, pid, row, fp, p))
    print(f"  ({tag}) {pid} -> {os.path.basename(fp)} (+svg)")

# optional: mirror the files into a working directory
if P_EXPORT:
    os.makedirs(P_EXPORT, exist_ok=True)
    for f in os.listdir(LOCAL):
        try:
            shutil.copy2(os.path.join(LOCAL, f), os.path.join(P_EXPORT, f))
            print("copied:", f)
        except Exception:
            print("copy failed (file in use):", f)
print("\noutput:", LOCAL)
