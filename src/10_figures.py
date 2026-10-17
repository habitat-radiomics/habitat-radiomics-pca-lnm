#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""10 — figures.

Every figure is drawn from the tables produced by ``src/05``–``src/09``; no
figure reads a per-patient score file directly.  That is deliberate: the
documented failure mode in the archive was a figure drawn from a superseded
score column (Fig. S4 showed 35/186/294 instead of 40/190/300).

Usage
-----
    python src/10_figures.py               # all figures
    python src/10_figures.py --only roc dca calibration shap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrpca import banner, ensure_dir, load_config, log, paths  # noqa: E402
from hrpca import metrics as M                                # noqa: E402
from hrpca.data import COHORTS, EXTERNAL                      # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

PALETTE = ["#1b4f72", "#c0392b", "#1e8449", "#7d3c98", "#b9770e",
           "#117a65", "#884ea0", "#5d6d7e"]
CA = ["#2166ac", "#b2182b"]

FIG_DPI = 600
FONT = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False}
plt.rcParams.update(FONT)


def _save(fig, out: Path, stem: str, dpi: int = FIG_DPI) -> None:
    """Save ``stem``.png and ``stem``.pdf, writing atomically.

    Writing straight to the destination can fail with ``OSError: [Errno 22]``
    when a viewer (or a cloud-sync agent) momentarily holds the previous file, so
    each figure is written to a temporary name and then moved into place, with a
    short retry.
    """
    import os
    import time

    for ext in ("png", "pdf"):
        dest = out / f"{stem}.{ext}"
        # the temporary name must keep the real extension: matplotlib infers the
        # output format from it
        tmp = out / f".{stem}.tmp.{ext}"
        fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
        for attempt in range(6):
            try:
                os.replace(tmp, dest)
                break
            except OSError as exc:
                if attempt == 5:
                    raise
                log(f"  retry saving {dest.name}: {exc}")
                time.sleep(0.5)


def _roc_curve(y, p):
    order = np.argsort(-np.asarray(p, dtype=float))
    y = np.asarray(y)[order]
    tpr = np.concatenate([[0.0], np.cumsum(y) / max(y.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / max((1 - y).sum(), 1)])
    return fpr, tpr


#: published cohort titles (Fig. 2 / 3a / 3b)
TITLES = {"train": "Training", "test1": "Test cohort 1", "test2": "Test cohort 2",
          "test_prospective": "Prospective"}

#: Fig. S7 row labels — the wording used by Table S2 panel (b)
ROW_LABELS = {"GLCM IDM": "GLCM inverse difference moment",
              "Entropy": "Entropy (heterogeneity)",
              "Signal mean": "Signal mean"}


def _training_scores(unrounded_path: Path, deployed_path: Path,
                     combined_cfg: dict | None = None) -> pd.DataFrame:
    """Training-cohort scores with the CORRECT source per column.

    Two independent source traps apply to the training cohort:

    1. The released training score file rounds probabilities to two decimals,
       which shifts the published training AUCs (clinical 0.704 → 0.706,
       all-habitat 0.863 → 0.864).  The unrounded 16-digit export is used for
       the five margin signatures.
    2. The ``Combined`` column of that historical export is a **superseded
       model** (AUC 0.795, not the published 0.798), so it is not used either.
       The published ``Combined`` is **recomputed from the paper's own
       coefficients** applied to the unrounded ``Clinic`` / ``HabitatH2`` /
       ``HabitatH4`` columns — which reproduces the published 0.798.

    ★ Pairing the two files positionally is NOT valid: they are in different
    row orders and the deployed file carries only a within-cohort index, so a
    positional mix silently produced Combined AUC 0.505.
    """
    up = pd.read_csv(unrounded_path, dtype=str, encoding="utf-8-sig")
    up = up[up["group"] == "train"].reset_index(drop=True)

    out = pd.DataFrame({"label": pd.to_numeric(up["label"], errors="coerce").values})
    for col in ("Clinic", "Habitat", "HabitatH1", "HabitatH2", "HabitatH3",
                "HabitatH4", "HabitatH5"):
        out[col] = pd.to_numeric(up[col], errors="coerce").values

    if combined_cfg:
        coef, b0 = combined_cfg["coef"], combined_cfg["intercept"]
        z = (b0 + coef[0] * out["Clinic"].values
             + coef[1] * out["HabitatH2"].values
             + coef[2] * out["HabitatH4"].values)
        out["Combined"] = 1.0 / (1.0 + np.exp(-z))
        a = M.auc(out["label"].values, out["Combined"].values)
        if round(a, 3) != 0.798:
            raise RuntimeError(
                f"recomputed training Combined AUC is {a:.6f}, expected 0.798 — "
                "check the coefficients and the unrounded source")
    else:
        out["Combined"] = pd.to_numeric(up["Combined_superseded"], errors="coerce").values
    return out


def fig_roc(scores, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    # label the clinical signature exactly as the paper does
    show = [("Clinic", "Clinical (ISUP + PI-RADS)"), ("HabitatH2", "Habitat H2"),
            ("HabitatH4", "Habitat H4"), ("Habitat", "All-habitat (27 features)"),
            ("Combined", "Combined")]
    for ax, coh in zip(axes, COHORTS):
        d = scores[coh]
        for i, (col, lab) in enumerate(show):
            fpr, tpr = _roc_curve(d.label.values, d[col].values)
            a = M.auc(d.label.values, d[col].values)
            lw = 2.2 if col == "Combined" else 1.2
            ax.plot(fpr, tpr, color=PALETTE[i], lw=lw, label=f"{lab} ({a:.3f})")
        ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity" if coh == "train" else "")
        ax.set_title(f"{TITLES[coh]} (n = {len(d)})", fontsize=9)
        ax.legend(fontsize=5.6, loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, out, "Fig2_ROC")
    plt.close(fig)


def fig_calibration(scores, pooled, out: Path, bins: int = 10):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, name in zip(axes, COHORTS + [None]):
        d = pooled if name is None else scores[name]
        title = "pooled external" if name is None else name
        for i, (col, lab) in enumerate([("Combined", "Combined"), ("Clinic", "Clinical")]):
            q = pd.qcut(d[col].values, bins, duplicates="drop")
            g = pd.DataFrame({"y": d.label.values, "p": d[col].values, "b": q}).groupby(
                "b", observed=True).agg(o=("y", "mean"), p=("p", "mean"))
            ax.plot(g.p, g.o, "o-", color=PALETTE[i], ms=3.5, lw=1.2, label=lab)
        ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8)
        p = M.hosmer_lemeshow(d.label.values, d.Combined.values, bins)
        ax.set_title(f"{title}\nH–L P = {p:.3f}", fontsize=9)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed proportion" if name == "train" else "")
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    _save(fig, out, "Fig3a_calibration")
    plt.close(fig)


def fig_dca(scores, pooled, out: Path, grid=(0.05, 0.90)):
    ts = np.linspace(grid[0], grid[1], 86)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, name in zip(axes, COHORTS + [None]):
        d = pooled if name is None else scores[name]
        title = "pooled external" if name is None else name
        y = d.label.values
        nb_c = [M.net_benefit(y, d.Combined.values, t) for t in ts]
        nb_l = [M.net_benefit(y, d.Clinic.values, t) for t in ts]
        nb_a = [M.net_benefit(y, np.ones(len(y)), t) for t in ts]
        ax.plot(ts, nb_c, color=PALETTE[0], lw=1.6, label="Combined")
        ax.plot(ts, nb_l, color=PALETTE[1], lw=1.2, label="Clinical")
        ax.plot(ts, nb_a, color="grey", ls="--", lw=0.9, label="Treat all")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylim(min(-0.02, min(nb_c) * 1.05), max(nb_c) * 1.15)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit" if name == "train" else "")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    _save(fig, out, "Fig3b_decision_curve")
    plt.close(fig)


def fig_shap(out: Path, cfg: dict):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hrpca.model import CombinedModel

    model = CombinedModel.from_config(cfg)
    tbl = pd.read_csv(out.parent / "tables" / "figS9_mean_abs_shap.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.barh(tbl["input"][::-1], tbl["mean_abs_shap"][::-1], color=PALETTE[0], height=0.6)
    for y, v in enumerate(tbl["mean_abs_shap"][::-1]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlabel("mean |SHAP|")
    ax.set_xlim(0, max(tbl["mean_abs_shap"]) * 1.25)
    fig.tight_layout()
    _save(fig, out, "FigS9_SHAP")
    plt.close(fig)


def fig_k_selection(out: Path, data_dir: Path):
    f = data_dir / "K_CV10x10_summary.csv"
    if not f.exists():
        log("  K_CV10x10_summary.csv not found — skipping Fig. S2")
        return
    d = pd.read_csv(f, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.errorbar(d["K"], d["best_per_fold_mean"], yerr=d["best_per_fold_SD"], fmt="o-",
                color=PALETTE[0], ms=4, lw=1.2, capsize=2)
    sel = d.loc[d["best_per_fold_mean"].idxmax()]
    ax.scatter([sel["K"]], [sel["best_per_fold_mean"]], color=PALETTE[1], zorder=5, s=32)
    ax.annotate(f"K = {int(sel['K'])}\nmean AUC {sel['best_per_fold_mean']:.3f}",
                xy=(sel["K"], sel["best_per_fold_mean"]),
                xytext=(sel["K"] + 0.4, sel["best_per_fold_mean"] - 0.02), fontsize=7)
    ax.axhline(0.5, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("K (habitats)"); ax.set_ylabel("Mean best-algorithm AUC\n(100 folds)")
    fig.tight_layout()
    _save(fig, out, "FigS2_K_selection")
    plt.close(fig)


def fig_habitat_profiles(out: Path, quantiles: pd.DataFrame):
    """Fig. S7 — habitat profile heatmap with the z-score printed in each cell.

    The figure is aligned with **Table S2 panel (b)**: the same 10 rows, in the
    table's own wording (``GLCM inverse difference moment``, not ``GLCM IDM``).
    The rows ``Kurtosis``, ``Skewness`` and ``GLCM cluster shade`` that appear in
    an earlier version of the published figure have **no authoritative source**
    in the submitted Table S2 and are therefore not drawn.

    The published figure annotates every cell, and so does this one; without the
    annotations the figure shows no numbers at all and cannot be compared.
    """
    if quantiles is None or quantiles.empty:
        log("  habitat profile table not found — skipping Fig. S7")
        return
    m = quantiles.set_index("feature")
    # defensive: the authoritative table's wording wins over any older label
    m = m.rename(index=ROW_LABELS)
    cols = [c for c in m.columns if c.startswith("H")]
    vals = m[cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(vals, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=6)
    # annotate each cell, in the paper's "+0.00" / "-0.00" style
    lim = max(abs(vals).max(), 1e-9)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if abs(v) > 0.6 * lim else "black")
    fig.colorbar(im, ax=ax, label="z-score across habitats", shrink=0.8)
    fig.tight_layout()
    _save(fig, out, "FigS7_habitat_profiles")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["roc", "calibration", "dca", "shap", "k", "profiles"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    out = P["results_figures"]
    ensure_dir(out)
    banner("10 · figures")
    log(f"  Arial font available: {'Arial' in {f.name for f in matplotlib.font_manager.fontManager.ttflist}}")

    # ★ two score sets are needed, because the published figures use different
    #   sources for different quantities:
    #     * Fig. 2's legend AUCs need the UNROUNDED export (clinical 0.706 rather
    #       than 0.704, all-habitat 0.864 rather than 0.863);
    #     * Fig. 3a/3b's calibration and decision curves need the DEPLOYED scores
    #       — their Hosmer–Lemeshow P (0.351) is only reproduced by the deployed
    #       Combined column, not by a recomputation (0.538).
    deployed = {c: pd.read_csv(P["data_scores"] / f"{c}.csv", encoding="utf-8-sig")
                for c in COHORTS}
    scores = dict(deployed)
    if P["data"].joinpath("scores_highprec.csv").exists():
        scores["train"] = _training_scores(P["data"] / "scores_highprec.csv",
                                           P["data_scores"] / "train.csv",
                                           cfg["model"]["combined"])
        log("  training cohort: 5 signatures from the unrounded export, "
            "Combined recomputed from the paper coefficients")
    pooled = pd.concat([deployed[c] for c in EXTERNAL], ignore_index=True)
    log("  calibration / DCA curves use the deployed scores (published H–L P)")
    only = args.only or ["roc", "calibration", "dca", "shap", "k", "profiles"]

    prof = None
    pf = P["results_tables"] / "tableS2b_habitat_zscores.csv"
    if pf.exists():
        prof = pd.read_csv(pf, encoding="utf-8-sig")

    if "roc" in only:
        fig_roc(scores, out); log("  Fig. 2 ROC")
    if "calibration" in only:
        fig_calibration(deployed, pooled, out, cfg["statistics"]["hosmer_lemeshow_bins"])
        log("  Fig. 3a calibration")
    if "dca" in only:
        fig_dca(deployed, pooled, out, cfg["statistics"]["dca_grid"]); log("  Fig. 3b decision curve")
    if "shap" in only:
        fig_shap(out, cfg); log("  Fig. S9 SHAP")
    if "k" in only:
        fig_k_selection(out, P["data"]); log("  Fig. S2 K selection")
    if "profiles" in only:
        fig_habitat_profiles(out, prof); log("  Fig. S7 habitat profiles")

    log(f"\n  figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
