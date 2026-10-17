#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06 — validation: AUC with bootstrap CIs, DeLong tests, agreement and robustness.

Reproduces:

* **Table 2** AUC with 95% bootstrap CIs (2000 resamples);
* **Table 4 / Table S5 / Table S12b** — DeLong P values for the combined model
  against each comparator, per cohort and pooled;
* **Table S12a** — external mean, population SD and range per signature;
* **Table S21 / Table 5** — subgroup AUCs and interaction P values
  (only when identifier linkage is available: ``--input-map``);
* **Table 2 (DL rows) / Table S19** — the deep-learning comparator, read from
  the frozen pooled results.

Usage
-----
    python src/06_validation.py
    python src/06_validation.py --bootstrap 2000
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

COMPARATORS = [("Clinical", "Clinic"), ("H4", "HabitatH4"),
               ("H2", "HabitatH2"), ("All-habitat", "Habitat")]


def load(ctx):
    scores = {c: pd.read_csv(ctx["data_scores"] / f"{c}.csv", encoding="utf-8-sig")
              for c in COHORTS}
    pooled = pd.concat([scores[c] for c in EXTERNAL], ignore_index=True)
    return scores, pooled


def table2_ci(scores, pooled, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for sig in SIGNATURES:
        row = {"signature": sig}
        for name, d in list(scores.items()) + [("pooled", pooled)]:
            y = d.label.values
            p = d[sig].values
            a = M.auc(y, p)
            lo, hi, _ = M.bootstrap_ci(y, p, n_boot=n_boot, seed=seed)
            row[f"{name}_auc"] = round(a, 4)
            row[f"{name}_lo"] = round(lo, 4)
            row[f"{name}_hi"] = round(hi, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def delong_table(scores, pooled) -> pd.DataFrame:
    rows = []
    for name, d in list(scores.items()) + [("pooled", pooled)]:
        row = {"cohort": name, "n": len(d),
               "auc_combined": round(M.auc(d.label.values, d.Combined.values), 4)}
        for label, col in COMPARATORS:
            p, dauc = M.delong_p(d.label.values, d.Combined.values, d[col].values)
            row[f"P_vs_{label}"] = round(p, 4)
            row[f"dAUC_vs_{label}"] = round(dauc, 4)
            row[f"auc_{label}"] = round(M.auc(d.label.values, d[col].values), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_pooled(pooled) -> pd.DataFrame:
    pairs = [("Combined", "HabitatH4"), ("Combined", "HabitatH2"), ("HabitatH4", "HabitatH2"),
             ("Combined", "Clinic"), ("HabitatH4", "Clinic"), ("HabitatH2", "Clinic"),
             ("Habitat", "Clinic"), ("Combined", "Habitat")]
    rows = []
    for a, b in pairs:
        p, dauc = M.delong_p(pooled.label.values, pooled[a].values, pooled[b].values)
        rows.append(dict(comparison=f"{a} vs {b}", auc_a=round(M.auc(pooled.label.values, pooled[a].values), 4),
                         auc_b=round(M.auc(pooled.label.values, pooled[b].values), 4),
                         dauc=round(dauc, 4), delong_p=round(p, 4)))
    return pd.DataFrame(rows)


def external_robustness(scores) -> pd.DataFrame:
    rows = []
    for sig in SIGNATURES:
        vals = [M.auc(scores[c].label.values, scores[c][sig].values) for c in EXTERNAL]
        rows.append(dict(signature=sig, train=round(M.auc(scores["train"].label.values,
                                                          scores["train"][sig].values), 4),
                         **{c: round(v, 4) for c, v in zip(EXTERNAL, vals)},
                         external_mean=round(float(np.mean(vals)), 4),
                         external_sd=round(float(np.std(vals, ddof=0)), 4),
                         external_range=round(float(max(vals) - min(vals)), 4)))
    return pd.DataFrame(rows)


def dl_table(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    return d


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    n_boot = args.bootstrap or cfg["statistics"]["auc_ci_bootstrap"]
    seed = cfg["statistics"]["random_state"]
    banner("06 · validation — bootstrap CIs, DeLong, robustness")
    scores, pooled = load(P)

    log("\n[1] Table 2 — AUC with 95% bootstrap CIs")
    t = table2_ci(scores, pooled, n_boot, seed)
    for _, r in t.iterrows():
        log(f"  {r.signature:<12} train {r.train_auc:.3f} "
            f"({r.train_lo:.3f}-{r.train_hi:.3f})  "
            f"test1 {r.test1_auc:.3f}  test2 {r.test2_auc:.3f}  "
            f"prospective {r.test_prospective_auc:.3f}  pooled {r.pooled_auc:.3f}")
    t.to_csv(P["results_tables"] / "table2_auc_ci.csv", index=False, encoding="utf-8-sig")

    log("\n[2] DeLong tests (combined vs each comparator)")
    dt = delong_table(scores, pooled)
    log(dt[["cohort", "n", "P_vs_Clinical", "P_vs_H4", "P_vs_H2", "P_vs_All-habitat"]]
        .to_string(index=False))
    dt.to_csv(P["results_tables"] / "table4_delong.csv", index=False, encoding="utf-8-sig")

    log("\n[3] pooled external pairwise comparisons (Table S12b)")
    pw = pairwise_pooled(pooled)
    log(pw.to_string(index=False))
    pw.to_csv(P["results_tables"] / "tableS12b_pairwise.csv", index=False, encoding="utf-8-sig")

    log("\n[4] external robustness per signature (Table S12a)")
    rb = external_robustness(scores)
    log(rb.to_string(index=False))
    rb.to_csv(P["results_tables"] / "tableS12a_robustness.csv", index=False, encoding="utf-8-sig")

    dlp = P["data"] / "dl_comparator_results.csv"
    if dlp.exists():
        log("\n[5] deep-learning comparator (frozen pooled results)")
        log(dl_table(dlp).to_string(index=False))

    write_json({"n_boot": n_boot, "seed": seed}, P["results"] / "validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
