#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""09 — SHAP explanation of the final combined model (closed form).

The combined model is linear in the three inputs, so the Shapley values have the
closed form ``phi_i = beta_i (x_i - E[x_i])`` and no sampling approximation is
needed.  ``E[x_i]`` is the **mean of the training cohort** (n = 398) — the same
cohort the published values were computed on.

Reproduces **Fig. 4 / Fig. S9** (mean |SHAP| = 0.521 clinical, 0.565 H4,
0.291 H2) and the per-patient force-plot values of **Fig. 6 / Fig. 8**,
including the additivity check (residual < 1e-15).

Usage
-----
    python src/09_shap.py
    python src/09_shap.py --cases S0043 S0036 S0797 S0959 S0968
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrpca import banner, load_config, log, paths, write_json  # noqa: E402
from hrpca import gold as G                                   # noqa: E402
from hrpca.model import CombinedModel                         # noqa: E402


def mean_abs_shap_table(model: CombinedModel, train: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per input, ordered by magnitude (Fig. 4a / Fig. S9a)."""
    vals = model.mean_abs_shap(train)
    rows = [dict(input=col, mean_abs_shap=round(v, 4)) for col, v in vals.items()]
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def per_patient_shap(model: CombinedModel, df: pd.DataFrame,
                     baseline: np.ndarray | None = None) -> pd.DataFrame:
    """Per-patient SHAP values plus the additivity check.

    ``baseline`` must be the training-cohort mean of each input; passing ``None``
    would centre on *df* itself and is only correct when *df* is the whole
    reference cohort.
    """
    if baseline is None:
        baseline = df.mean(axis=0)
    shap, base = model.shap_values(df, baseline=baseline)
    out = pd.DataFrame(shap, columns=[f"phi_{f}" for f in model.features], index=df.index)
    out["phi_sum"] = shap.sum(axis=1)
    out["baseline"] = float(model.intercept)
    out["f_x_reconstructed"] = out["baseline"] + out["phi_sum"]
    if "f_x" in df.columns:
        out["f_x_stored"] = df["f_x"].values
        out["residual"] = out["f_x_reconstructed"] - out["f_x_stored"]
    out["baseline_vector"] = ",".join(f"{b:.6f}" for b in base)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cases", nargs="*", default=None, help="patient IDs for the force plots")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    banner("09 · SHAP explanation of the final combined model (closed form)")
    model = CombinedModel.from_config(cfg)
    log(f"  model: logit(p) = {model.intercept:.6f} + "
        + " + ".join(f"{b:.6f}*{f}" for f, b in zip(model.features, model.coef)))

    # ---- training cohort, full 398 patients ------------------------------
    hi = pd.read_csv(P["data"] / "scores_highprec.csv", encoding="utf-8-sig", dtype={"ID": str})
    # the 16-digit export names the inputs HabitatH2 / HabitatH4, the model
    # uses the canonical names; resolve_column handles the mapping
    train = hi[hi.group == "train"].copy()
    tbl = mean_abs_shap_table(model, train)
    log("\n  [1] mean |SHAP| on the training cohort (n = 398)")
    log(tbl.to_string(index=False))
    paper = {"Clinic": G.SHAP_MEAN_ABS["clinical"], "HabitatH2": G.SHAP_MEAN_ABS["H2"],
             "HabitatH4": G.SHAP_MEAN_ABS["H4"]}
    worst = 0.0
    for _, r in tbl.iterrows():
        p = paper.get(r["input"])
        if p is not None:
            worst = max(worst, abs(p - r.mean_abs_shap))
    log(f"  max |recomputed − published| = {worst:.4f}")
    tbl.to_csv(P["results_tables"] / "figS9_mean_abs_shap.csv", index=False, encoding="utf-8-sig")

    # ---- per-patient values and additivity -------------------------------
    meta_path = P["data"] / "shap" / "meta_all.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, encoding="utf-8-sig")
        # baseline = training-cohort mean of each input (n = 398)
        base_vec = np.array([float(train[c].mean()) for c in
                             ["Clinic", "HabitatH2", "HabitatH4"]])
        pp = per_patient_shap(model, meta, baseline=base_vec)
        pp.to_csv(P["results_tables"] / "shap_per_patient.csv", index=True, encoding="utf-8-sig")
        # additivity is exact for the linear model: sum(phi) + f(baseline) == f(x)
        f_base = model.intercept + float(np.dot(model.coef, base_vec))
        pp["f_x_reconstructed"] = f_base + pp["phi_sum"]
        pp["residual"] = pp["f_x_reconstructed"] - pp["f_x_stored"]
        pp.to_csv(P["results_tables"] / "shap_per_patient.csv", index=True, encoding="utf-8-sig")
        log("\n  [2] additivity check on meta_all.csv (n = %d)" % len(meta))
        log(f"      f(baseline) = {f_base:.7f}")
        log(f"      max |sum(phi) + f(baseline) − stored f(x)| = {pp['residual'].abs().max():.2e}")
        base = float(pd.read_csv(P["data"] / "shap" / "base.csv", encoding="utf-8-sig").iloc[0, 0])
        log(f"      shipped baseline (mean f(x) over all rows) = {base:.7f}")
        log(f"      mean f(x) over the 391 training rows        = "
            f"{model.raw_logit(meta[meta.group == 'train']).mean():.7f}")

        cases = args.cases or G.FIG6_CASES
        log("\n  [3] the five illustrated cases (Fig. 6 / Fig. 8)")
        ids = meta.copy()
        ids["_id"] = ids["ID"].astype(str)
        for cid in cases:
            hit = ids[ids["_id"] == cid]
            if hit.empty:
                log(f"      {cid}: not found")
                continue
            r = hit.iloc[0]
            phi, _ = model.shap_values(hit, baseline=base_vec)
            log(f"      {cid}  {r.group:<18} pN{r.label}  p={1 / (1 + np.exp(-r.f_x)):.3f}  "
                f"f(x)={r.f_x:+.3f}  phi=" +
                "  ".join(f"{c} {v:+.4f}" for c, v in zip(model.features, phi[0])))
    else:
        log("  meta_all.csv not found — skipping the per-patient panel")

    write_json({"model": {"features": list(model.features), "coef": list(model.coef),
                          "intercept": model.intercept},
                "mean_abs_shap": tbl.set_index("input")["mean_abs_shap"].to_dict(),
                "convention": "E[x] = mean of the training cohort (n = 398); "
                              "phi_i = beta_i (x_i - E[x_i])"},
               P["results"] / "shap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
