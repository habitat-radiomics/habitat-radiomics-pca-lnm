#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05 — model training and the frozen combined model.

Reproduces, from the released per-patient scores:

* **Table 2 / Table S7** — the AUC of every signature in every cohort;
* the **pooled external** AUCs and the external mean / SD / range
  (Table S12a);
* the **combined model** coefficients and the baseline E[f(X)]
  (``data/shap/base.csv``);
* the **equivalent nomogram parameterisation** (ISUP + PI-RADS entered
  separately instead of the clinical probability);
* the **habitat-number comparison** at K = 4 / 5 / 7 (Table 3 / Table S3b),
  using the K-specific frozen exports.

Two source conventions matter and are enforced here (see
``docs/OLD_CODE_INVENTORY.md``):

1. training-cohort AUCs come from the **16-significant-digit** export
   (``data/scores_highprec.csv``), never from the two-decimal
   ``data/scores/train.csv``;
2. the **combined** model's training AUC comes from the deployed scores in
   ``data/scores/train.csv``; the ``Combined`` column of the historical
   high-precision export belongs to a superseded model (0.795 vs 0.798).

Usage
-----
    python src/05_model_training.py                 # all of the above
    python src/05_model_training.py --k-comparison  # only Table 3 / S3b
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
from hrpca.data import COHORTS, EXTERNAL, SIGNATURES          # noqa: E402
from hrpca.model import CombinedModel, fit_logistic           # noqa: E402


# ---------------------------------------------------------------------------
def table2(P: dict) -> pd.DataFrame:
    hi = pd.read_csv(P["data"] / "scores_highprec.csv", encoding="utf-8-sig")
    scores = {c: pd.read_csv(P["data_scores"] / f"{c}.csv", encoding="utf-8-sig") for c in COHORTS}
    pooled = pd.concat([scores[c] for c in EXTERNAL], ignore_index=True)
    tr_hi = hi[hi.group == "train"]
    rows = []
    for sig in SIGNATURES:
        row = {"signature": sig}
        for c in COHORTS:
            if c == "train" and sig != "Combined":
                val = M.auc(tr_hi.label.values, tr_hi[sig].values)
            else:
                d = scores[c]
                val = M.auc(d.label.values, d[sig].values)
            row[c] = round(val, 4)
        row["pooled_external"] = round(M.auc(pooled.label.values, pooled[sig].values), 4)
        ext = [row[c] for c in EXTERNAL]
        row["external_mean"] = round(float(np.mean(ext)), 4)
        row["external_sd"] = round(float(np.std(ext, ddof=0)), 4)
        row["external_range"] = round(float(max(ext) - min(ext)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def table3_k(cfg: dict, P: dict, clinical: pd.DataFrame) -> pd.DataFrame:
    valid = set(clinical["ID"].astype(str))
    rows = []
    for k, spec in cfg["habitat_partitions"].items():
        f = P["data"] / "k_runs" / f"K{k}.csv"
        if not f.exists():
            log(f"  K = {k}: {f.name} missing — skipped")
            continue
        d = pd.read_csv(f, encoding="utf-8-sig")
        retained = spec["retained"]
        if "row_index" in d.columns:
            aucs = {c: M.auc(d[d.cohort == c].label.values, d[d.cohort == c]["Combined"].values)
                    for c in COHORTS}
            note = "frozen Combined column"
        else:
            d["_id"] = d["ID"].astype(str)
            d = d[d["_id"].isin(valid)]
            m = d.merge(clinical[["ID", "ISUP", "PI_RADS"]].rename(columns={"ID": "_id"}),
                        on="_id", how="left")
            m = m.dropna(subset=["ISUP", "PI_RADS"] + retained)
            tr = m[m.cohort == "train"]
            if tr.empty:
                log(f"  K = {k}: no training rows after the identifier merge — skipped")
                continue
            lr1 = fit_logistic(tr[["ISUP", "PI_RADS"]], tr.label)
            m = m.copy()
            m["ClinicNew"] = lr1.predict_proba(m[["ISUP", "PI_RADS"]])[:, 1]
            tr = m[m.cohort == "train"]
            lr2 = fit_logistic(tr[["ClinicNew"] + retained], tr.label)
            m["Comb"] = lr2.predict_proba(m[["ClinicNew"] + retained])[:, 1]
            aucs = {c: M.auc(m[m.cohort == c].label.values, m[m.cohort == c].Comb.values)
                    for c in COHORTS}
            note = "refit from K-specific scores"
        ext = [aucs[c] for c in EXTERNAL]
        rows.append(dict(K=k, retained="+".join(r.replace("Habitat", "") for r in retained),
                         train=round(aucs["train"], 4),
                         test1=round(aucs["test1"], 4), test2=round(aucs["test2"], 4),
                         test_prospective=round(aucs["test_prospective"], 4),
                         mean_external=round(float(np.mean(ext)), 4),
                         decay=round(aucs["train"] - float(np.mean(ext)), 4),
                         source=note))
    return pd.DataFrame(rows)


def combine_scores(d: pd.DataFrame, model: CombinedModel) -> np.ndarray:
    return model.predict(d)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--k-comparison", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    clinical = pd.read_csv(P["data"] / "clinical.csv", encoding="utf-8-sig", dtype={"ID": str})
    banner("05 · model training and the frozen combined model")

    if not args.k_comparison:
        log("\n[1] Table 2 / Table S7 — signature AUCs")
        t2 = table2(P)
        log(t2.to_string(index=False))
        t2.to_csv(P["results_tables"] / "table2_signature_aucs.csv", index=False,
                  encoding="utf-8-sig")

        log("\n[2] combined model coefficients and baseline")
        model = CombinedModel()
        log(f"    logit(p) = {model.intercept:.6f} + "
            + " + ".join(f"{b:.6f}*{f}" for f, b in zip(model.features, model.coef)))
        sh = pd.read_csv(P["data"] / "shap" / "meta_all.csv", encoding="utf-8-sig")
        tr = sh[sh.group == "train"]
        z = model.raw_logit(tr)
        log(f"    baseline E[f(X)] over the 391-row SHAP subset: {z.mean():.6f}")
        log(f"    shipped base.csv                          : "
            f"{float(pd.read_csv(P['data'] / 'shap' / 'base.csv', encoding='utf-8-sig').iloc[0, 0]):.6f}")
        write_json({"features": model.features, "coef": model.coef,
                    "intercept": model.intercept,
                    "baseline_shap_subset": float(z.mean())},
                   P["results"] / "combined_model.json")

        log("\n[3] equivalent nomogram parameterisation (ISUP + PI-RADS separate)")
        hi = pd.read_csv(P["data"] / "model_inputs.csv", encoding="utf-8-sig", dtype={"ID": str})
        trn = hi[hi.group == "train"].dropna(subset=["ISUP", "PI_RADS", "H2", "H4"])
        lr = fit_logistic(trn[["ISUP", "PI_RADS", "H2", "H4"]], trn.label)
        p = lr.predict_proba(trn[["ISUP", "PI_RADS", "H2", "H4"]])[:, 1]
        log(f"    coefficients: {np.round(lr.coef_[0], 6).tolist()}  intercept {lr.intercept_[0]:.6f}")
        log(f"    training AUC: {M.auc(trn.label.values, p):.6f}  (paper: 0.795)")

    log("\n[4] Table 3 / Table S3b — habitat number K")
    t3 = table3_k(cfg, P, clinical)
    log(t3.to_string(index=False))
    t3.to_csv(P["results_tables"] / "table3_habitat_number.csv", index=False,
              encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
