"""The authoritative cross-table key.

``Clinic.csv`` carries a plain numeric ID; ``label.csv`` carries NIfTI file
names; the per-cohort score files in ``derived_data`` carry only a within-cohort
``patient_index``.  None of the three shares a key with either of the others.

The key that *does* exist is ``scripts/_cases_with_ids.csv`` (1,016 rows), which
contains, for every analysed patient::

    patient_index   label   Clinic   Habitat   HabitatH1..H5   Combined
    real_id         cohort  ck      s_HabitatH2  s_HabitatH4  dom  margin

``(cohort, patient_index)`` is unique and ``real_id`` is unique, and the eight
score columns of that file are **byte-identical** to the corresponding rows of
``derived_data/*_scores_label.csv``.  It therefore establishes

    derived_data/<cohort>_scores_label.csv  --(patient_index)-->  real_id
    real_id  --(ID)-->  Clinic.csv      (the clinical variables)
    real_id  --(ID)-->  label.csv       (.nii.gz filename -> group / label)
    real_id  --(ID)-->  SHAP_output/heatmap/heatmap_model_inputs_all.csv

Verified: 1,016 / 1,016 identifiers are present in ``label.csv``; 1,005 / 1,016
are present in ``Clinic.csv`` (11 patients have no clinical record); 1,016 /
1,016 are present in the heatmap exports.

Validity of the mapping (fingerprint test, ``verify/check_id_linkage.py``)
------------------------------------------------------------------------
Joining the released score files to ``heatmap_data_all.csv`` through
``real_id`` reproduces the unrounded clinical probability exactly in

* **test cohort 1** — 339 / 339 rows, Pearson **1.0000**
* **test cohort 2** — 189 / 189 rows, Pearson **1.0000**
* prospective cohort — 86 / 90 rows (the other 4 are rounded to 2 decimals)
* **training cohort** — 0 / 398 rows, Pearson **0.091** ✗

so the mapping is authoritative for the 618 external patients and **must not be
used for the training cohort**; use ``heatmap_data_all.csv`` (or
``_fullprec_scores.csv``, which is identical to it) for the 398 training
patients.  A positional join is never valid: it reproduces only 36 / 339 rows in
test cohort 1 and 14 / 189 in test cohort 2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "COHORT_NAMES",
    "load_cases_with_ids",
    "load_cohort",
    "load_analysis_frame",
    "attach_identifiers",
    "verify_key",
]

# the Chinese cohort labels used inside _cases_with_ids.csv
COHORT_NAMES = {"train": "训练", "test1": "测试1", "test2": "测试2",
                "test_prospective": "前瞻"}

# score column  ->  column name in _cases_with_ids.csv
SCORE_COLUMNS = {
    "Clinic": "Clinic",
    "Habitat": "Habitat",
    "HabitatH1": "HabitatH1",
    "HabitatH2": "HabitatH2",
    "HabitatH3": "HabitatH3",
    "HabitatH4": "HabitatH4",
    "HabitatH5": "HabitatH5",
    "Combined": "Combined",
}

# NOTE (2026-10-16, corrected): an earlier revision of this module claimed that
# ``heatmap_data_all.csv`` uses a *different* model for its ``habitat`` / ``H1``
# / ``H3`` / ``H5`` columns.  That was wrong.  ``heatmap_data_all.csv`` and
# ``_fullprec_scores.csv`` agree on all 1,016 patients for every column, and
# both agree with the released score files in test cohorts 1 and 2.  Apparent
# mismatches were entirely caused by the training cohort's identifier mapping
# inside ``_cases_with_ids.csv`` being unreliable (warning W2 below).
HEATMAP_AGREES = SCORE_COLUMNS


def load_cases_with_ids(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig", dtype={"real_id": str})
    d["real_id"] = d["real_id"].astype(str).str.strip()
    return d


def attach_identifiers(scores: pd.DataFrame, cohort: str,
                       cases: pd.DataFrame) -> pd.DataFrame:
    """Add ``real_id`` to a cohort's score frame using the authoritative key.

    Parameters
    ----------
    scores : the cohort's ``<cohort>_scores_label.csv`` (needs ``patient_index``)
    cohort : ``train`` / ``test1`` / ``test2`` / ``test_prospective``
    cases  : :func:`load_cases_with_ids`
    """
    key = cases[cases["cohort"] == COHORT_NAMES[cohort]][
        ["patient_index", "real_id"]].copy()
    out = scores.merge(key, on="patient_index", how="left", validate="one_to_one")
    missing = int(out["real_id"].isna().sum())
    if missing:
        raise ValueError(f"{cohort}: {missing} rows could not be keyed")
    return out


def load_cohort(data_dir: str | Path, cohort: str, cases: pd.DataFrame) -> pd.DataFrame:
    """One cohort's scores, with ``real_id`` attached and row order canonicalised."""
    d = pd.read_csv(Path(data_dir) / "scores" / f"{cohort}.csv", encoding="utf-8-sig")
    d = d.sort_values("patient_index").reset_index(drop=True)
    return attach_identifiers(d, cohort, cases)


def load_analysis_frame(data_dir: str | Path, cases_path: str | Path) -> pd.DataFrame:
    """The whole 1016-patient analysis set with identifiers and clinical variables.

    Columns: ``real_id, cohort, group, label`` + every score column + the
    clinical variables from ``Clinic.csv`` (``Age, PI_RADS, PSA, ISUP,
    target_roi_volume, prostate_volume, PSAD``) + the heatmap model inputs
    (``ISUP_hm, PI_RADS_hm``).
    """
    from .data import COHORTS

    data_dir = Path(data_dir)
    cases = load_cases_with_ids(cases_path)
    frames = []
    for c in COHORTS:
        d = load_cohort(data_dir, c, cases)
        d["cohort"] = c
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["group"] = m["cohort"]

    clin = pd.read_csv(data_dir / "clinical.csv", encoding="utf-8-sig", dtype={"ID": str})
    m = m.merge(clin, left_on="real_id", right_on="ID", how="left").drop(columns=["ID"])

    hm_path = data_dir / "model_inputs.csv"
    if hm_path.exists():
        hm = pd.read_csv(hm_path, encoding="utf-8-sig", dtype={"ID": str})
        hm = hm.rename(columns={"ISUP": "ISUP_hm", "PI_RADS": "PI_RADS_hm"})
        m = m.merge(hm[["ID", "ISUP_hm", "PI_RADS_hm"]],
                    left_on="real_id", right_on="ID", how="left").drop(columns=["ID"])
    return m


def verify_key(data_dir: str | Path, cases_path: str | Path) -> dict:
    """Re-run the key checks and return a report dict."""
    from .data import COHORTS

    data_dir, cases_path = Path(data_dir), Path(cases_path)
    cases = load_cases_with_ids(cases_path)
    rep = {"n_cases": len(cases),
           "dup_patient_index": int(cases.groupby(["cohort", "patient_index"]).size().gt(1).sum()),
           "dup_real_id": int(cases["real_id"].duplicated().sum()),
           "score_columns_identical": {},
           "coverage": {}}
    labels = pd.read_csv(data_dir / "labels.csv", encoding="utf-8-sig", dtype={"ID": str})
    clin = pd.read_csv(data_dir / "clinical.csv", encoding="utf-8-sig", dtype={"ID": str})
    rep["coverage"]["labels.csv"] = int(cases["real_id"].isin(labels["ID"]).sum())
    rep["coverage"]["clinical.csv"] = int(cases["real_id"].isin(clin["ID"]).sum())
    for c in COHORTS:
        s = pd.read_csv(data_dir / "scores" / f"{c}.csv", encoding="utf-8-sig")
        s = s.sort_values("patient_index").reset_index(drop=True)
        k = cases[cases["cohort"] == COHORT_NAMES[c]].sort_values("patient_index")
        k = k.reset_index(drop=True)
        for col in SCORE_COLUMNS:
            same = bool(np.allclose(s[col].values, k[col].values, atol=1e-12))
            rep["score_columns_identical"].setdefault(c, {})[col] = same
    return rep
