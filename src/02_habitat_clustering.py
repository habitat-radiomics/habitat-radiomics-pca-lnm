#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02 — two-level K-means habitat decomposition and habitat-number selection.

Level 1 (whole gland)  : voxel-wise K-means on ADC, DWI and normalised T2WI
                         within the gland; K = 2 by the Calinski-Harabasz index;
                         the cluster with the lowest mean ADC is the target
                         region.
Level 2 (target region): voxel-wise K-means on the local radiomics feature
                         space; K = 5 selected by repeated 10x10-fold
                         cross-validation on the clinical endpoint within the
                         training cohort (Table S3).

Requires voxel data (not distributed).  This module exposes:

* :func:`level1_target_region` and :func:`level2_habitats` — the two clustering
  steps, with the exact conventions used in the study;
* :func:`internal_criteria` — Calinski-Harabasz, silhouette and Davies-Bouldin
  by K (Table S3c);
* :func:`k_selection_report` — the ranked K table and the selected K
  (Table S3 / Fig. S2), read from ``data/K_CV10x10_summary.csv``;
* :func:`tiny_habitat_report` — proportion of patients with a habitat below the
  50-voxel threshold (Table S4 / Fig. S1).

Usage
-----
    # K selection report from the frozen 100-fold summary
    python src/02_habitat_clustering.py --report

    # internal criteria on a voxel feature matrix (npy: rows = voxels)
    python src/02_habitat_clustering.py --criteria voxels.npy --k 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrpca import banner, load_config, log, paths, write_json  # noqa: E402

TINY_VOXELS = 50          # < 50 voxels ~ 38 mm^3 at 0.5 x 0.5 x 3 mm
LEVEL1_K = 2
LEVEL1_SELECTION = "Calinski-Harabasz index"
LEVEL2_SELECTION = "highest mean best-algorithm AUC over 10 repeats x 10 folds"


# ---------------------------------------------------------------------------
# level 1 — whole-gland intensity clustering
# ---------------------------------------------------------------------------
def level1_target_region(voxels: np.ndarray, k: int = LEVEL1_K, seed: int = 0):
    """Cluster voxels on (ADC, DWI, T2WI); return the lowest-ADC cluster label.

    ``voxels`` is ``(n_voxels, 3)`` with columns ``[ADC, DWI, T2WI_normalised]``.
    The target region is defined by a **within-gland relative ranking** (lowest
    mean ADC), not by an absolute ADC threshold, so a scanner-dependent offset
    in absolute ADC does not change the partition.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(voxels)
    adc = voxels[:, 0]
    means = [adc[km.labels_ == c].mean() for c in range(k)]
    return int(np.argmin(means)), km


# ---------------------------------------------------------------------------
# level 2 — intratumoral texture clustering
# ---------------------------------------------------------------------------
def level2_habitats(features: np.ndarray, k: int = 5, seed: int = 0):
    """K-means over the local radiomics feature space inside the target region."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(features)


def internal_criteria(features: np.ndarray, ks) -> pd.DataFrame:
    """Calinski-Harabasz, silhouette and Davies-Bouldin for each K (Table S3c)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    rows = []
    for k in ks:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=0).fit_predict(features)
        rows.append(dict(k=int(k),
                         calinski_harabasz=float(calinski_harabasz_score(features, km)),
                         silhouette=float(silhouette_score(features, km)),
                         davies_bouldin=float(davies_bouldin_score(features, km))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# K selection from the frozen 100-fold summary
# ---------------------------------------------------------------------------
def k_selection_report(data_dir: Path) -> pd.DataFrame:
    f = Path(data_dir) / "K_CV10x10_summary.csv"
    d = pd.read_csv(f, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    d = d.sort_values("K").reset_index(drop=True)
    d["rank"] = d["best_per_fold_mean"].rank(ascending=False, method="min").astype(int)
    return d


def tiny_habitat_report(partitions: dict) -> pd.DataFrame:
    """``{K: {patient_id: label_map}}`` -> proportion with a habitat < 50 voxels."""
    rows = []
    for k, per_patient in sorted(partitions.items()):
        n = len(per_patient)
        tiny = 0
        for labels in per_patient.values():
            counts = np.bincount(np.asarray(labels, dtype=int))
            counts = counts[1:] if counts.size > 1 and counts[0] == 0 else counts
            if counts.size and counts.min() < TINY_VOXELS:
                tiny += 1
        rows.append(dict(k=int(k), n=n, n_tiny=tiny,
                         pct=100.0 * tiny / n if n else float("nan")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--report", action="store_true", help="K selection report (Table S3 / Fig. S2)")
    ap.add_argument("--criteria", default=None, help="npy file: voxels x features")
    ap.add_argument("--k", nargs="*", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9])
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    banner("02 · two-level K-means habitat decomposition")

    if args.report or not args.criteria:
        d = k_selection_report(P["data"])
        log("\n  Table S3 — habitat-number selection (100 folds per K)")
        log(d[["K", "n_folds", "best_algo", "best_per_fold_mean", "best_per_fold_SD",
               "all_algo_mean", "rank"]].to_string(index=False))
        best = int(d.loc[d["best_per_fold_mean"].idxmax(), "K"])
        log(f"\n  selected K = {best}  ({LEVEL2_SELECTION})")
        log(f"  config says K = {cfg['k_selection']['selected']}")
        ok = best == cfg["k_selection"]["selected"]
        log(f"  {'OK' if ok else 'MISMATCH'}")
        d.to_csv(P["results_tables"] / "tableS3_k_selection.csv", index=False, encoding="utf-8-sig")
        if not args.criteria:
            return 0 if ok else 1

    if args.criteria:
        feats = np.load(args.criteria)
        log(f"\n  voxel feature matrix: {feats.shape}")
        crit = internal_criteria(feats, args.k)
        log("\n  Table S3c — internal clustering criteria")
        log(crit.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        crit.to_csv(P["results_tables"] / "tableS3c_internal_criteria.csv",
                    index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
