#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""08 — decision-curve analysis and its translation into clinical terms.

Reproduces **Table S13** and **Fig. S4**, and the decision-impact panels of
**Table S17(b)/(c)**.

Definitions
-----------
* net benefit      ``NB(t) = TP/n - FP/n * t/(1-t)``
* "unnecessary ePLND avoided" — the pN0 patients below the threshold
* "pN1 missed"      — the pN1 patients below the threshold
* recalibrated panel: Cox recalibration fitted within the pooled external cohorts

Usage
-----
    python src/08_dca.py
    python src/08_dca.py --thresholds 0.05 0.10 0.15 0.20 0.25 0.30
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
from hrpca.data import COHORTS, EXTERNAL                      # noqa: E402


def clinical_terms(d: pd.DataFrame, threshold: float) -> dict:
    y = d.label.values
    n = len(y)
    n1 = int((y == 1).sum())
    n0 = n - n1
    out = {}
    for tag, col in (("combined", "Combined"), ("clinical", "Clinic")):
        pos = d[col].values >= threshold
        tp = int(((y == 1) & pos).sum())
        fp = int(((y == 0) & pos).sum())
        tn = int(((y == 0) & ~pos).sum())          # pN0 below threshold
        out[tag] = dict(
            classified_high_risk=int(pos.sum()),
            eplnd=int(pos.sum()),
            avoided_unnecessary=tn,                # truly avoided ePLND (was fp)
            missed=n1 - tp,
            sensitivity=tp / n1 if n1 else float("nan"),
            specificity=tn / n0 if n0 else float("nan"),   # was fp / n0 (that is 1 - spec)
            net_benefit=M.net_benefit(y, d[col].values, threshold),
        )
    out["n"] = n
    out["pn1"] = n1
    out["pn0"] = n0
    return out


def table_s13(scores, pooled, thresholds) -> pd.DataFrame:
    rows = []
    for name, d in list(scores.items()) + [("pooled", pooled)]:
        for t in thresholds:
            ct = clinical_terms(d, t)
            rows.append(dict(
                cohort=name, threshold=t,
                eplnd=ct["combined"]["eplnd"],
                avoided=ct["combined"]["avoided_unnecessary"],
                missed=ct["combined"]["missed"],
                sensitivity=f"{ct['combined']['sensitivity'] * 100:.1f}%",
                specificity=f"{ct['combined']['specificity'] * 100:.1f}%",
                net_benefit=round(ct["combined"]["net_benefit"], 4),
                clinical_eplnd=ct["clinical"]["eplnd"],
                clinical_avoided=ct["clinical"]["avoided_unnecessary"],
                clinical_missed=ct["clinical"]["missed"],
                clinical_net_benefit=round(ct["clinical"]["net_benefit"], 4),
            ))
    return pd.DataFrame(rows)


def table_s17bc(pooled, thresholds) -> tuple:
    y = pooled.label.values
    a, b, pr = M.cox_recalibration(y, pooled.Combined.values)
    a2, b2, prc = M.cox_recalibration(y, pooled.Clinic.values)
    rows_nb, rows_imp = [], []
    for t in thresholds:
        rows_nb.append(dict(
            threshold=t,
            combined_original=round(M.net_benefit(y, pooled.Combined.values, t), 4),
            combined_recalibrated=round(M.net_benefit(y, pr, t), 4),
            clinical_original=round(M.net_benefit(y, pooled.Clinic.values, t), 4),
            clinical_recalibrated=round(M.net_benefit(y, prc, t), 4),
        ))
        for tag, p in (("combined", pr), ("clinical", prc)):
            pos = p >= t
            tp = int(((y == 1) & pos).sum())
            tn = int(((y == 0) & ~pos).sum())      # pN0 below threshold
            rows_imp.append(dict(scale="recalibrated", threshold=t, model=tag,
                                 classified=int(pos.sum()), missed=int((y == 1).sum()) - tp,
                                 avoided=tn))
    return pd.DataFrame(rows_nb), pd.DataFrame(rows_imp)


def dca_grid(pooled, grid) -> pd.DataFrame:
    ts = np.linspace(grid[0], grid[1], 86)
    y = pooled.label.values
    rows = []
    for t in ts:
        rows.append(dict(
            threshold=float(t),
            combined=M.net_benefit(y, pooled.Combined.values, t),
            clinical=M.net_benefit(y, pooled.Clinic.values, t),
            all_patients=M.net_benefit(y, np.ones(len(y)), t),
            none=0.0,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--thresholds", nargs="*", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    thresholds = args.thresholds or cfg["statistics"]["dca_thresholds"]
    banner("08 · decision-curve analysis and clinical terms")

    scores = {c: pd.read_csv(P["data_scores"] / f"{c}.csv", encoding="utf-8-sig") for c in COHORTS}
    pooled = pd.concat([scores[c] for c in EXTERNAL], ignore_index=True)

    log(f"\n[1] Table S13 — clinical terms at thresholds {thresholds}")
    t13 = table_s13(scores, pooled, thresholds)
    show = t13[t13.cohort == "pooled"]
    log(show[["threshold", "eplnd", "avoided", "missed", "sensitivity",
              "specificity", "net_benefit", "clinical_avoided", "clinical_missed"]]
        .to_string(index=False))
    t13.to_csv(P["results_tables"] / "tableS13_dca_clinical_terms.csv", index=False,
               encoding="utf-8-sig")

    log("\n[2] Table S17(b)/(c) — decision impact and net benefit after recalibration")
    nb, imp = table_s17bc(pooled, thresholds)
    log("  net benefit:"); log(nb.to_string(index=False))
    log("  recalibrated decision impact:"); log(imp.to_string(index=False))
    nb.to_csv(P["results_tables"] / "tableS17c_net_benefit.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(P["results_tables"] / "tableS17b_decision_impact.csv", index=False,
               encoding="utf-8-sig")

    log("\n[3] net-benefit grid for Fig. 3b / Fig. S4")
    g = dca_grid(pooled, cfg["statistics"]["dca_grid"])
    g.to_csv(P["results_tables"] / "dca_grid_pooled.csv", index=False, encoding="utf-8-sig")
    log(f"  {len(g)} thresholds from {g.threshold.min():.2f} to {g.threshold.max():.2f}")

    write_json({"thresholds": thresholds, "grid": cfg["statistics"]["dca_grid"],
                "cohorts": COHORTS}, P["results"] / "dca.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
