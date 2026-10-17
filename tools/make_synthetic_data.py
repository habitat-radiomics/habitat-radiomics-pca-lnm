#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a **fully synthetic** dataset with the same schema as the real one.

Why
---
The individual-level analysis dataset cannot be shared (ChiCTR2500111002:
*IPD sharing: No*).  A synthetic dataset with identical column names and file
layout lets a reader run the whole pipeline end-to-end without any real patient
record.

★ Every value written by this script is FABRICATED from a seeded random number
  generator.  Nothing is copied, resampled, permuted or otherwise derived from
  the real data: the real tables are used **only** to read the column names, and
  the control flow never reads a real cell value.

★ Never claim anything measured on these files is a study result.  They exist to
  exercise the code path, not to reproduce the paper (see README).

Usage
-----
    python tools/make_synthetic_data.py            # write data/synthetic/
    python tools/make_synthetic_data.py --n 40     # change the cohort size
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: where the real tables live (OUTSIDE the public repository)
CONTROLLED = Path(r"C:\Users\l\Desktop\lunwen3\00_受控数据_禁传\repo_data_个体级")

OUT = REPO / "data" / "synthetic"
COHORTS = ["train", "test1", "test2", "test_prospective"]
SIGNATURES = ["Clinic", "Habitat", "HabitatH1", "HabitatH2", "HabitatH3",
              "HabitatH4", "HabitatH5", "Combined"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _study_ids(n: int) -> list:
    return [f"SYN{i:04d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
def clinical(n: int, rng) -> pd.DataFrame:
    """Same columns as ``data/clinical.csv``: ID, Age, PI_RADS, PSA, ISUP,
    target_roi_volume, prostate_volume, PSAD."""
    return pd.DataFrame({
        "ID": _study_ids(n),
        "Age": rng.integers(50, 80, n).astype(float),
        "PI_RADS": rng.integers(2, 6, n),
        "PSA": np.round(rng.gamma(2.0, 8.0, n), 1),
        "ISUP": rng.integers(1, 6, n).astype(float),
        "target_roi_volume": np.round(rng.gamma(3.0, 5.0, n), 3),
        "prostate_volume": np.round(rng.gamma(8.0, 4.5, n), 3),
        "PSAD": None,
    }).assign(PSAD=lambda d: np.round(d["PSA"] / d["prostate_volume"], 6))


def labels(n: int, rng) -> pd.DataFrame:
    """Same columns as ``data/labels.csv``: ID, label, group."""
    ids = _study_ids(n)
    g = (["train"] * int(n * 0.5) + ["test1"] * int(n * 0.25)
         + ["test2"] * int(n * 0.15) + ["test_prospective"] * int(n * 0.1))
    g += [g[-1]] * (n - len(g))
    return pd.DataFrame({"ID": ids, "label": rng.integers(0, 2, n), "group": g[:n]})


def analysis_set(ids, rng) -> pd.DataFrame:
    """Same columns, in the same order, as ``data/analysis_set.csv``."""
    return pd.DataFrame({"ID": ids,
                         "group": ["train"] * len(ids),
                         "label": rng.integers(0, 2, len(ids))})


def model_inputs(ids, rng) -> pd.DataFrame:
    """Same columns as ``data/model_inputs.csv``: ID, group, label, ISUP, PI_RADS, H2, H4."""
    k = len(ids)
    return pd.DataFrame({
        "ID": ids, "group": ["train"] * k, "label": rng.integers(0, 2, k),
        "ISUP": rng.integers(1, 6, k), "PI_RADS": rng.integers(2, 6, k),
        "H2": np.round(rng.random(k), 16), "H4": np.round(rng.random(k), 16)})


def scores(ids, rng, n_cohorts: int = 4) -> dict:
    """Same structure as ``data/scores/<cohort>.csv``:
    patient_index, label, Clinic, Habitat, HabitatH1..H5, Combined."""
    out = {}
    per = max(1, len(ids) // n_cohorts)
    for i, c in enumerate(COHORTS):
        sl = ids[i * per:(i + 1) * per] if i < n_cohorts - 1 else ids[i * per:]
        k = len(sl)
        if not k:
            continue
        d = pd.DataFrame({"patient_index": np.arange(1, k + 1),
                          "label": rng.integers(0, 2, k)})
        for s in SIGNATURES:
            d[s] = np.round(rng.random(k), 16)
        out[c] = d
    return out


def cases_with_ids(scores_by_cohort) -> pd.DataFrame:
    """Same columns as ``data/cases_with_ids.csv``."""
    rows = []
    for c, d in scores_by_cohort.items():
        for _, r in d.iterrows():
            row = {"patient_index": int(r["patient_index"]), "label": int(r["label"])}
            for s in SIGNATURES:
                row[s] = float(r[s])
            row.update({"real_id": f"SYN{int(r['patient_index']):04d}", "cohort": c,
                        "ck": "—", "s_HabitatH2": 0.0, "s_HabitatH4": 0.0,
                        "dom": "H2", "margin": 0.0})
            rows.append(row)
    return pd.DataFrame(rows)


def feature_table(n: int, h: int, columns: list, rng) -> pd.DataFrame:
    """Same columns as ``data/features/rad_features_h<h>.csv`` — 1,197 features.

    ``ID`` carries the ``.nii.gz`` suffix, exactly as the real tables do.
    """
    d = pd.DataFrame({"ID": [f"SYN{i:04d}.nii.gz" for i in range(1, n + 1)]})
    for c in columns:
        d[c] = np.round(rng.random(n), 10)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of synthetic patients")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--features", action="store_true",
                    help="also write the (large) 1,197-column feature tables")
    args = ap.parse_args()

    rng = _rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    n = args.n
    cl = clinical(n, rng)
    lb = labels(n, rng)
    cl.to_csv(OUT / "clinical.csv", index=False, encoding="utf-8-sig")
    lb.to_csv(OUT / "labels.csv", index=False, encoding="utf-8-sig")
    ids = cl["ID"].tolist()
    analysis_set(ids, rng).to_csv(OUT / "analysis_set.csv", index=False, encoding="utf-8-sig")
    model_inputs(ids, rng).to_csv(OUT / "model_inputs.csv", index=False, encoding="utf-8-sig")

    sc = scores(ids, rng)
    (OUT / "scores").mkdir(exist_ok=True)
    for c, d in sc.items():
        d.to_csv(OUT / "scores" / f"{c}.csv", index=False, encoding="utf-8-sig")
    cases_with_ids(sc).to_csv(OUT / "cases_with_ids.csv", index=False, encoding="utf-8-sig")

    written = ["clinical.csv", "labels.csv", "analysis_set.csv", "model_inputs.csv",
               "cases_with_ids.csv"] + [f"scores/{c}.csv" for c in sc]

    # feature tables: identical column names (read from the controlled copy),
    # entirely fabricated values
    if args.features:
        (OUT / "features").mkdir(exist_ok=True)
        for h in range(1, 6):
            src = CONTROLLED / "features" / f"rad_features_h{h}.csv"
            if not src.exists():
                print(f"  !! cannot read the column names of {src.name} — skipping")
                continue
            cols = list(pd.read_csv(src, nrows=0, encoding="utf-8-sig").columns)
            cols = [c for c in cols if c != "ID"]
            feature_table(n, h, cols, rng).to_csv(
                OUT / "features" / f"rad_features_h{h}.csv", index=False, encoding="utf-8-sig")
            written.append(f"features/rad_features_h{h}.csv")

    (OUT / "README.md").write_text(
        "# SYNTHETIC DATA — for pipeline demonstration only\n\n"
        "**Not patient data. Not derived from patient data. Not study results.**\n\n"
        "Every value in this directory was generated by a seeded random number\n"
        "generator (`tools/make_synthetic_data.py`, seed 20260101).  The real\n"
        "tables were consulted **only** to copy column names; no real cell value\n"
        "was read into this dataset.\n\n"
        "The individual-level analysis dataset is not publicly available\n"
        "(ChiCTR2500111002: *IPD sharing: No*); see the top-level README.\n\n"
        "Purpose: let a reader run the code end-to-end and confirm that the\n"
        "pipeline executes.  **Numbers computed on these files are meaningless** —\n"
        "they will not, and must not, match the paper.\n\n"
        "| file | rows | mirrors |\n|---|---|---|\n"
        f"| `clinical.csv` | {n} | `data/clinical.csv` (columns identical) |\n"
        f"| `labels.csv` | {n} | `data/labels.csv` |\n"
        f"| `analysis_set.csv` | {n} | `data/analysis_set.csv` |\n"
        f"| `model_inputs.csv` | {n} | `data/model_inputs.csv` |\n"
        f"| `cases_with_ids.csv` | {sum(len(d) for d in sc.values())} | `data/cases_with_ids.csv` |\n"
        f"| `scores/*.csv` | {n} total | `data/scores/<cohort>.csv` |\n"
        + (f"| `features/rad_features_h1..5.csv` | {n} each | `data/features/rad_features_h*.csv` (1,197 columns) |\n"
           if args.features else ""),
        encoding="utf-8")

    print(f"synthetic cohort: n = {n}, seed = {args.seed}")
    for w in written:
        p = OUT / w
        print(f"  {w:<38} {p.stat().st_size:>9,} B")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
