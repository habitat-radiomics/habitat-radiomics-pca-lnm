"""Data loading.

Inputs (all under ``data/``) are produced by ``src/00_assemble_data.py``:

    data/clinical.csv           ID, Age, PI_RADS, PSA, ISUP, target_roi_volume,
                                prostate_volume, PSAD             (n = 1097 raw)
    data/labels.csv             ID, label, group                 (n = 1125 raw)
    data/analysis_set.csv       ID, group, label                  (n = 1016, frozen)
    data/scores/<cohort>.csv    label + one column per signature  (frozen predictions)
    data/scores/README...       see data/README.md
    data/features/rad_features_h{1..5}.csv                       (habitat feature tables)

The frozen analysis set is defined by ``data/analysis_set.csv``; every other
table is restricted to those identifiers.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "COHORTS",
    "EXTERNAL",
    "SIGNATURES",
    "SIGNATURE_COLUMNS",
    "load_labels",
    "load_clinical",
    "load_analysis_set",
    "load_scores",
    "load_scores_long",
    "load_heatmap_inputs",
    "load_features",
    "load_k_results",
]

COHORTS = ["train", "test1", "test2", "test_prospective"]
EXTERNAL = ["test1", "test2", "test_prospective"]

SIGNATURES = [
    "Clinic",
    "Habitat",
    "HabitatH1",
    "HabitatH2",
    "HabitatH3",
    "HabitatH4",
    "HabitatH5",
    "Combined",
]

# canonical table row order used by every results table in the paper
SIGNATURE_COLUMNS = {s: s for s in SIGNATURES}


# --------------------------------------------------------------------------
def _id_str(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(".nii.gz", "", regex=False)
        .str.replace(".nii", "", regex=False)
        .str.strip()
    )


def load_labels(data_dir) -> pd.DataFrame:
    """``ID, label, group`` — the raw screening table (1125 rows)."""
    d = pd.read_csv(Path(data_dir) / "labels.csv", encoding="utf-8-sig", dtype={"ID": str})
    d["ID"] = _id_str(d["ID"])
    return d


def load_clinical(data_dir) -> pd.DataFrame:
    """De-identified clinical variables (1097 rows, *no* imputation)."""
    d = pd.read_csv(Path(data_dir) / "clinical.csv", encoding="utf-8-sig", dtype={"ID": str})
    d["ID"] = _id_str(d["ID"])
    return d


def load_analysis_set(data_dir) -> pd.DataFrame:
    """The frozen 1016-patient analysis set: ``ID, group, label``."""
    d = pd.read_csv(Path(data_dir) / "analysis_set.csv", encoding="utf-8-sig", dtype={"ID": str})
    d["ID"] = _id_str(d["ID"])
    return d


def load_scores(data_dir) -> dict:
    """``{cohort: DataFrame}`` with the frozen per-signature predictions.

    Row order inside each file is the order in which the files were exported
    and is *not* the cohort order of ``analysis_set.csv``; every row-level
    statistic in this repository is order-invariant, and no analysis joins
    these files to identifiers.
    """
    out = {}
    base = Path(data_dir) / "scores"
    for c in COHORTS:
        d = pd.read_csv(base / f"{c}.csv", encoding="utf-8-sig")
        out[c] = d
    return out


def load_scores_long(data_dir) -> pd.DataFrame:
    """Concatenate the five cohort files with a ``cohort`` column."""
    frames = []
    for c, d in load_scores(data_dir).items():
        d = d.copy()
        d["cohort"] = c
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_heatmap_inputs(data_dir) -> pd.DataFrame:
    """``ID, group, label, ISUP, PI_RADS, H2, H4, Combined`` (n = 1016).

    This is the high-precision (16 significant digit) table: it is the only
    source used for the *training* AUCs and for anything requiring unrounded
    probabilities.  The ``Combined`` column of the historical heatmap export
    belonged to a superseded model and is therefore **not** used; the deployed
    combined-model scores come from ``data/scores/``.
    """
    return pd.read_csv(Path(data_dir) / "model_inputs.csv", encoding="utf-8-sig")


def load_features(data_dir, k: int = 5) -> dict:
    """``{habitat_index: DataFrame}`` of the habitat-level radiomics tables."""
    out = {}
    for h in range(1, k + 1):
        f = Path(data_dir) / "features" / f"rad_features_h{h}.csv"
        if f.exists():
            out[h] = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
    return out


def load_k_results(data_dir, k: int) -> pd.DataFrame:
    """Frozen K-specific export: ``ID, label, Clinic, Habitat…, Combined``."""
    return pd.read_csv(Path(data_dir) / "k_runs" / f"K{k}.csv", encoding="utf-8-sig")
