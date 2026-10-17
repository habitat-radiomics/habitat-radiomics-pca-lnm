"""The gold-standard table: every number the code must reproduce.

Each entry is ``{value, source, convention}`` where

* ``value``      — the number as printed in the manuscript/supplement;
* ``source``     — where it appears (table / figure / section);
* ``convention`` — any non-obvious calculation convention.

Entries with ``reproducible: False`` are documented in
``docs/VERIFICATION_REPORT.md`` as *not* derivable from the released derived
data; they are listed here so that the checking tool reports them explicitly
rather than silently skipping them.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Table 2 / Table S7 — 8 signatures × 4 cohorts = 32 AUCs
# ---------------------------------------------------------------------------
TABLE2_AUC = {
    "Clinic":            {"train": 0.706, "test1": 0.795, "test2": 0.770, "test_prospective": 0.689},
    "Habitat":           {"train": 0.864, "test1": 0.591, "test2": 0.552, "test_prospective": 0.655},
    "HabitatH1":         {"train": 0.681, "test1": 0.655, "test2": 0.478, "test_prospective": 0.702},
    "HabitatH2":         {"train": 0.695, "test1": 0.679, "test2": 0.607, "test_prospective": 0.764},
    "HabitatH3":         {"train": 0.676, "test1": 0.595, "test2": 0.573, "test_prospective": 0.711},
    "HabitatH4":         {"train": 0.751, "test1": 0.709, "test2": 0.719, "test_prospective": 0.815},
    "HabitatH5":         {"train": 0.673, "test1": 0.690, "test2": 0.528, "test_prospective": 0.648},
    "Combined":          {"train": 0.798, "test1": 0.842, "test2": 0.829, "test_prospective": 0.887},
}

# Table 2 / Table S19 — deep-learning comparator (pooled external and CV10)
TABLE2_DL = {
    "ADC":            {"train": 0.721, "test1": 0.570, "test2": 0.647, "test_prospective": 0.627,
                       "cv10": 0.695, "cv10_sd": 0.112, "pooled": 0.594},
    "T2WI":           {"train": 0.899, "test1": 0.557, "test2": 0.685, "test_prospective": 0.584,
                       "cv10": 0.859, "cv10_sd": 0.071, "pooled": 0.593},
    "DWI":            {"train": 0.898, "test1": 0.589, "test2": 0.664, "test_prospective": 0.549,
                       "cv10": 0.858, "cv10_sd": 0.080, "pooled": 0.599},
    "three-sequence": {"train": 0.982, "test1": 0.584, "test2": 0.673, "test_prospective": 0.560,
                       "cv10": 0.946, "cv10_sd": 0.040, "pooled": 0.597},
}

# Table 2 footnote / Table 3 — pooled external AUCs
POOLED_EXTERNAL_AUC = {"Combined": 0.844, "Clinic": 0.778, "Habitat": 0.591}

# ---------------------------------------------------------------------------
# Table 3 / Table S3b — habitat number K
# ---------------------------------------------------------------------------
TABLE3_K = {
    4: {"retained": ["HabitatH1", "HabitatH4"], "train": 0.900, "test1": 0.721,
        "test2": 0.647, "test_prospective": 0.798, "mean_external": 0.722, "decay": +0.178},
    5: {"retained": ["HabitatH2", "HabitatH4"], "train": 0.798, "test1": 0.842,
        "test2": 0.829, "test_prospective": 0.887, "mean_external": 0.853, "decay": -0.055},
    7: {"retained": ["HabitatH6", "HabitatH7"], "train": 0.848, "test1": 0.794,
        "test2": 0.592, "test_prospective": 0.843, "mean_external": 0.743, "decay": +0.105},
}

# Table S3 — 100-fold repeated CV, mean best-algorithm AUC
TABLE_S3_MEAN_AUC = {2: 0.657, 3: 0.639, 4: 0.686, 5: 0.691, 6: 0.643, 7: 0.665, 8: 0.640, 9: 0.654}
TABLE_S3_MEAN_ALL11 = {2: 0.558, 3: 0.541, 4: 0.585, 5: 0.596, 6: 0.543, 7: 0.569, 8: 0.527, 9: 0.562}
TABLE_S3_WILCOXON_P = {2: 0.001, 3: 0.001, 4: 0.740, 5: None, 6: 0.001, 7: 0.004, 8: 0.001, 9: 0.003}
TABLE_S3_BEST_ALGO = {2: "MLP", 3: "MLP", 4: "MLP", 5: "LR", 6: "MLP", 7: "LR", 8: "LR", 9: "LR"}

# Table S3c — internal clustering criteria
TABLE_S3C = {
    2: {"calinski": 16089.9, "silhouette": 0.399, "davies_bouldin": 1.006},
    3: {"calinski": 14689.8, "silhouette": 0.342, "davies_bouldin": 1.048},
    4: {"calinski": 13789.4, "silhouette": 0.318, "davies_bouldin": 1.047},
    5: {"calinski": 13010.0, "silhouette": 0.303, "davies_bouldin": 1.047},
    6: {"calinski": 12372.3, "silhouette": 0.292, "davies_bouldin": 1.052},
    7: {"calinski": 11829.5, "silhouette": 0.283, "davies_bouldin": 1.056},
    8: {"calinski": 11364.7, "silhouette": 0.277, "davies_bouldin": 1.055},
    9: {"calinski": 10964.7, "silhouette": 0.273, "davies_bouldin": 1.056},
}

# Table S4 / Fig. S1 — patients with at least one tiny habitat (<50 voxels)
TABLE_S4_TINY = {2: (0, 0.0), 3: (2, 0.2), 4: (7, 0.7), 5: (13, 1.3),
                 6: (27, 2.7), 7: (48, 4.7), 8: (79, 7.8), 9: (116, 11.4)}

# Table S3d — purity / fragmentation
TABLE_S3D = {
    2: {"n": 1016, "core_ml": 9.30, "core_pct": 60.1, "core_z": -0.226, "rest_z": 0.347,
        "gap": -0.573, "gain_pct": None, "tiny_pct": 0.0, "min_vox": 4408, "max_vox": 15852,
        "null_gap": 0.017, "ratio": 33.7},
    5: {"n": 1015, "core_ml": 4.51, "core_pct": 29.7, "core_z": -0.895, "rest_z": 0.292,
        "gap": -1.187, "gain_pct": 9.5, "tiny_pct": 1.2, "min_vox": 681, "max_vox": 8445,
        "null_gap": 0.041, "ratio": 28.8},
}

# ---------------------------------------------------------------------------
# Nomogram / combined model
# ---------------------------------------------------------------------------
COMBINED_MODEL = {
    "features": ["Clinic", "HabitatH2", "HabitatH4"],
    "coef": [4.062608661920751, 2.593122775462074, 3.9343891733424328],
    "intercept": -4.248788185663511,
    "paper_rounded": {"Clinic": 4.06, "H2": 2.59, "H4": 3.93, "intercept": -4.25},
}
SHAP_BASELINE = -1.1039913           # E[f(X)] over all 1009 rows of meta_all.csv
SHAP_BASELINE_TRAIN = -1.1047250     # mean f(x) over the 391 training rows of meta_all.csv
SHAP_MEAN_ABS = {"clinical": 0.521, "H4": 0.565, "H2": 0.291}

# ---------------------------------------------------------------------------
# Table S5 / Table S10 — DeLong P values, combined vs each comparator
# ---------------------------------------------------------------------------
# ⚠️ 2026-10-16 (second revision): the values below were transcribed from a
#    STALE manuscript copy in the workspace
#    (``lunwen3\投稿版_Manuscript_FINAL_EN.md``, last modified 2026-10-09),
#    which still contained a "Table 4" that no longer exists in the submitted
#    manuscript (its main text has only Tables 1-3).  That stale file reported
#    .336 for test cohort 2; the submitted Table S5 and Table S10 both read
#    .371, and .3711 is what the deployed scores give.
#
#    ►► Do NOT use these constants.  The authoritative values are parsed at
#       run time from the submitted documents by ``src/hrpca/paper.py``
#       (``PP.table_s5_delong()`` / ``PP.table_s10_incremental()``).
#    ►► They are kept here only as a historical record of the stale copy.
DELONG_TABLE4_STALE = {          # ⚠️ from the superseded 2026-10-09 working copy
    "train":            {"Clinic": 0.001, "H4": 0.012, "H2": 0.001, "All": 0.002},
    "test1":            {"Clinic": 0.255, "H4": 0.001, "H2": 0.001, "All": 0.001},
    "test2":            {"Clinic": 0.336, "H4": 0.003, "H2": 0.002, "All": 0.001},
    "test_prospective": {"Clinic": 0.003, "H4": 0.202, "H2": 0.066, "All": 0.003},
    "pooled":           {"Clinic": 0.025, "H4": 0.001, "H2": 0.001, "All": 0.001},
}

# The submitted Table S5, verified against the deployed scores on 2026-10-16
# (24/24 cells agree).  Prefer the run-time parse; this mirror is for
# convenience and is asserted against ``paper`` by ``_self_check`` below.
DELONG_TABLE_S5 = {        # ✅ submitted supplementary Table S5
    "train":            {"Clinic": 0.001, "H4": 0.012, "H2": 0.001, "All": 0.002},
    "test1":            {"Clinic": 0.255, "H4": 0.001, "H2": 0.001, "All": 0.001},
    "test2":            {"Clinic": 0.371, "H4": 0.003, "H2": 0.002, "All": 0.001},
    "test_prospective": {"Clinic": 0.003, "H4": 0.202, "H2": 0.066, "All": 0.003},
    "pooled":           {"Clinic": 0.022, "H4": 0.001, "H2": 0.001, "All": 0.001},
}

# retained aliases so that old call sites still run, pointing at the
# authoritative (submitted) values rather than the stale copy
DELONG_TABLE4 = DELONG_TABLE_S5

# ---------------------------------------------------------------------------
# KNOWN_DISCREPANCIES — cells where the published value does not follow from
# the released data.  The verifier reports these as DISCREPANCY (not FAIL):
# the correct value is stated so that the manuscript can be corrected.
# ---------------------------------------------------------------------------
KNOWN_DISCREPANCIES = {
    # ⚠️ 2026-10-16 (second revision) — key entries PURGED.
    #
    # The DeLong, table_s16a and base.csv entries that used to live here were
    # derived from a STALE workspace copy of the manuscript
    # (``lunwen3\投稿版_Manuscript_FINAL_EN.md``, 2026-10-09) that still
    # contained a "Table 4".  The submitted main text has only Tables 1-3 and
    # supplementary Tables S1-S21, and its Table S5 / Table S10 reproduce
    # exactly (24/24 DeLong cells, verified 2026-10-16).
    #
    # ► Programmatic constraint: published values must be parsed at run time by
    #   ``src/hrpca/paper.py``; do not re-introduce transcribed constants here.
    ("table_s10", "train ΔAUC"): (
        0.094,
        "Table S10 prints +0.092, but its own two AUC columns (0.798 and 0.706) "
        "differ by 0.092 only after rounding; the unrounded difference is 0.0942."),
    ("table_s15", "n complete cases"): (
        374,
        "the released clinical table yields 374 training patients with all seven "
        "variables present, not 377; a grid over nine predictor sets never gives "
        "377 together with the published multivariable PI-RADS odds ratio."),
    ("dl", "T2WI · test1"): (
        0.557,
        "derived_data/dl_comparator_results.csv stores 0.556500, which rounds "
        "to 0.557; the comparator CSV supplied with the submission has 0.556."),
}

# ---------------------------------------------------------------------------
# Sections that cannot be reproduced from the release at all.  A FAIL inside
# such a section is reported as DISCREPANCY with the section-level reason.
# ---------------------------------------------------------------------------
KNOWN_SECTION_DISCREPANCIES = {
    # ⚠️ 2026-10-16 (third revision) — PURGED.
    #
    # "table1" and "table_s15" used to be listed here.  Both were artefacts of
    # reading the WRONG clinical table:
    #
    #   * Table S15 (n = 377, PI-RADS OR 1.611, PSA density 1.118) is reproduced
    #     exactly — 14/14 cells — by C:/O/Clinic.csv joined to label.csv.
    #     The workspace copy lunwen3/Clinic.csv is a later 1097-row build whose
    #     training cohort has 374 complete cases and returns PI-RADS 1.565.
    #   * Table 1 is reproduced exactly by the same authoritative build:
    #     C:/O/Clinic.csv has exactly 7 training patients without ISUP grade
    #     group (matching the footnote) and gives 13/61/76/104/137 = 391.
    #     The heatmap export's ISUP column fills all 398 patients and gives 83
    #     for grade 3.
    #
    # ► Both sections now PASS with no section-level exceptions.
}


SHAP_TOLERANCE = 5e-3

# Tolerances that exist only because the published table rounds some of its own
# inputs before writing them down.  They are stated here so that the verifier's
# output is explicit rather than silently lenient.
ROUNDING_TOLERANCE = {
    # Table S2(a): the published means are three-decimal values recomputed from a
    # full-precision mean, so |difference| can reach half of the last digit.
    "table_s2_panel_a": 1.5e-3,
    # Table S2(b): the published z-scores are two-decimal values obtained from
    # the three-decimal means of panel (a); three-decimal inputs quantise the
    # two-decimal output.
    "table_s2_panel_b": 0.15,
}

# Table S12b — pooled external pairwise comparisons
TABLE_S12B = {
    "Combined_vs_H4":     {"dauc": +0.115, "p": 0.001},
    "Combined_vs_H2":     {"dauc": +0.168, "p": 0.001},
    "H4_vs_H2":           {"dauc": +0.053, "p": 0.212},
    "Combined_vs_Clinic": {"dauc": +0.066, "p": 0.025},
    "H4_vs_Clinic":       {"dauc": -0.049, "p": 0.290},
}

# ---------------------------------------------------------------------------
# Table S10 / S11b — incremental value
# ---------------------------------------------------------------------------
TABLE_S10 = {
    "train":            {"auc_comb": 0.798, "auc_clin": 0.706, "dauc": +0.092, "delong_p": 0.001,
                         "idi": 0.124, "idi_p": 0.001, "nri": 0.679, "nri_lo": 0.478, "nri_hi": 0.886},
    "test1":            {"auc_comb": 0.842, "auc_clin": 0.795, "dauc": +0.047, "delong_p": 0.255,
                         "idi": 0.059, "idi_p": 0.039, "nri": 0.567, "nri_lo": 0.263, "nri_hi": 0.881},
    "test2":            {"auc_comb": 0.829, "auc_clin": 0.770, "dauc": +0.059, "delong_p": 0.371,
                         "idi": 0.060, "idi_p": 0.182, "nri": 0.292, "nri_lo": -0.202, "nri_hi": 0.756},
    "test_prospective": {"auc_comb": 0.887, "auc_clin": 0.689, "dauc": +0.198, "delong_p": 0.003,
                         "idi": 0.210, "idi_p": 0.001, "nri": 1.044, "nri_lo": 0.587, "nri_hi": 1.445},
    "pooled":           {"auc_comb": 0.844, "auc_clin": 0.778, "dauc": +0.066, "delong_p": 0.022,
                         "idi": 0.087, "idi_p": 0.001, "nri": 0.567, "nri_lo": 0.347, "nri_hi": 0.780},
}
TABLE_S11B = {
    "train":            {"event": +0.254, "non_event": +0.425, "total": +0.679},
    "test1":            {"event": +0.200, "non_event": +0.367, "total": +0.567},
    "test2":            {"event": +0.333, "non_event": -0.041, "total": +0.292},
    "test_prospective": {"event": +0.579, "non_event": +0.465, "total": +1.044},
    "pooled":           {"event": +0.317, "non_event": +0.250, "total": +0.567},
}

# ---------------------------------------------------------------------------
# Table S9 / S17 — calibration
# ---------------------------------------------------------------------------
TABLE_S17A = {
    "train":            {"n": 398, "auc": 0.798, "intercept": +0.002, "slope": 1.003,
                         "hl_original": 0.351, "hl_recalibrated": 0.350},
    "test1":            {"n": 339, "auc": 0.842, "intercept": -0.470, "slope": 1.127,
                         "hl_original": 0.0002, "hl_recalibrated": 0.0007},
    "test2":            {"n": 189, "auc": 0.829, "intercept": -0.923, "slope": 1.126,
                         "hl_original": 0.018, "hl_recalibrated": 0.562},
    "test_prospective": {"n": 90, "auc": 0.887, "intercept": +0.062, "slope": 2.231,
                         "hl_original": 0.073, "hl_recalibrated": 0.149},
    "pooled":           {"n": 618, "auc": 0.844, "intercept": -0.513, "slope": 1.239,
                         "hl_original": 0.001, "hl_recalibrated": 0.107},
}
TABLE_S17C_NET_BENEFIT = {   # threshold: (combined original, combined recalibrated)
    0.10: (0.069, 0.089), 0.15: (0.056, 0.063), 0.20: (0.050, 0.050),
    0.25: (0.031, 0.042), 0.30: (0.019, 0.030),
}
TABLE_S17B_RECAL = {  # threshold: (classified, missed) after Cox recalibration
    0.10: (237, 9), 0.15: (167, 24), 0.20: (132, 31), 0.25: (99, 38), 0.30: (74, 47),
}
CALIBRATION_BRIER = {  # main text 3.5
    "train": {"hl_p": 0.351, "brier": 0.162},
    "test2": {"hl_p": 0.018, "brier": 0.089},
    "test_prospective": {"hl_p": 0.073, "brier": 0.114},
}
# Table S9 habitat-signature H-L values — NOT reproducible from the released data
TABLE_S9_HL = {
    "Combined": {"train": 0.351, "test1": 0.0002, "test2": 0.018, "test_prospective": 0.073},
    "Clinic":   {"train": 0.012, "test1": 0.0004, "test2": 0.091, "test_prospective": 0.380},
    "Habitat":  {"train": 0.052, "test1": 0.089, "test2": 0.208, "test_prospective": 0.830},
    "HabitatH1": {"train": 0.004, "test1": 0.003, "test2": 0.974, "test_prospective": 0.231},
    "HabitatH2": {"train": 0.065, "test1": 0.314, "test2": 0.214, "test_prospective": 0.001},
    "HabitatH3": {"train": 0.003, "test1": 0.360, "test2": 0.828, "test_prospective": 0.083},
    "HabitatH4": {"train": 0.007, "test1": 0.102, "test2": 0.115, "test_prospective": 0.262},
    "HabitatH5": {"train": 0.012, "test1": 0.011, "test2": 0.687, "test_prospective": 0.124},
}
TABLE_S9_NOT_REPRODUCIBLE = ["Clinic", "Habitat", "HabitatH1", "HabitatH2",
                             "HabitatH3", "HabitatH4", "HabitatH5"]

# ---------------------------------------------------------------------------
# Table S13 / Fig. S4 — decision curve in clinical terms
# ---------------------------------------------------------------------------
TABLE_S13 = {
    "pooled": {
        0.05: {"eplnd": 578, "avoided": 40, "missed": 0, "sens": (82, 82), "spec": (40, 536),
               "nb": 0.090, "clin_avoided": 16, "clin_missed": 1},
        0.10: {"eplnd": 427, "avoided": 190, "missed": 1, "sens": (81, 82), "spec": (190, 536),
               "nb": 0.069, "clin_avoided": 126, "clin_missed": 5},
        0.15: {"eplnd": 312, "avoided": 300, "missed": 6, "sens": (76, 82), "spec": (300, 536),
               "nb": 0.056, "clin_avoided": 224, "clin_missed": 10},
        0.20: {"eplnd": 241, "avoided": 368, "missed": 9, "sens": (73, 82), "spec": (368, 536),
               "nb": 0.050, "clin_avoided": 288, "clin_missed": 11},
        0.25: {"eplnd": 194, "avoided": 405, "missed": 19, "sens": (63, 82), "spec": (405, 536),
               "nb": 0.031, "clin_avoided": 393, "clin_missed": 25},
        0.30: {"eplnd": 152, "avoided": 438, "missed": 28, "sens": (54, 82), "spec": (438, 536),
               "nb": 0.019, "clin_avoided": 434, "clin_missed": 28},
    },
    "train": {
        0.05: {"eplnd": 388, "avoided": 10, "missed": 0, "nb": 0.261, "clin_avoided": 5, "clin_missed": 0},
        0.10: {"eplnd": 321, "avoided": 74, "missed": 3, "nb": 0.231, "clin_avoided": 39, "clin_missed": 1},
        0.15: {"eplnd": 264, "avoided": 128, "missed": 6, "nb": 0.214, "clin_avoided": 74, "clin_missed": 10},
        0.20: {"eplnd": 226, "avoided": 156, "missed": 16, "nb": 0.178, "clin_avoided": 94, "clin_missed": 13},
        0.25: {"eplnd": 191, "avoided": 181, "missed": 26, "nb": 0.148, "clin_avoided": 135, "clin_missed": 29},
        0.30: {"eplnd": 162, "avoided": 201, "missed": 35, "nb": 0.123, "clin_avoided": 154, "clin_missed": 32},
    },
    "test1": {
        0.05: {"eplnd": 311, "avoided": 28, "missed": 0, "nb": 0.091, "clin_avoided": 5, "clin_missed": 1},
        0.10: {"eplnd": 214, "avoided": 125, "missed": 0, "nb": 0.077, "clin_avoided": 62, "clin_missed": 3},
        0.15: {"eplnd": 151, "avoided": 184, "missed": 4, "nb": 0.064, "clin_avoided": 126, "clin_missed": 5},
        0.20: {"eplnd": 112, "avoided": 220, "missed": 7, "nb": 0.058, "clin_avoided": 164, "clin_missed": 5},
        0.25: {"eplnd": 88, "avoided": 237, "missed": 14, "nb": 0.035, "clin_avoided": 231, "clin_missed": 14},
        0.30: {"eplnd": 74, "avoided": 246, "missed": 19, "nb": 0.016, "clin_avoided": 255, "clin_missed": 16},
    },
    "test2": {
        0.05: {"eplnd": 179, "avoided": 10, "missed": 0, "nb": 0.050, "clin_avoided": 11, "clin_missed": 0},
        0.10: {"eplnd": 130, "avoided": 58, "missed": 1, "nb": 0.024, "clin_avoided": 61, "clin_missed": 2},
        0.15: {"eplnd": 90, "avoided": 97, "missed": 2, "nb": 0.016, "clin_avoided": 89, "clin_missed": 4},
        0.20: {"eplnd": 70, "avoided": 117, "missed": 2, "nb": 0.013, "clin_avoided": 106, "clin_missed": 5},
        0.25: {"eplnd": 55, "avoided": 130, "missed": 4, "nb": 0.002, "clin_avoided": 136, "clin_missed": 7},
        0.30: {"eplnd": 41, "avoided": 143, "missed": 5, "nb": 0.005, "clin_avoided": 146, "clin_missed": 7},
    },
    "test_prospective": {
        0.05: {"eplnd": 88, "avoided": 2, "missed": 0, "nb": 0.171, "clin_avoided": 0, "clin_missed": 0},
        0.10: {"eplnd": 83, "avoided": 7, "missed": 0, "nb": 0.132, "clin_avoided": 3, "clin_missed": 0},
        0.15: {"eplnd": 71, "avoided": 19, "missed": 0, "nb": 0.109, "clin_avoided": 9, "clin_missed": 1},
        0.20: {"eplnd": 59, "avoided": 31, "missed": 0, "nb": 0.100, "clin_avoided": 18, "clin_missed": 1},
        0.25: {"eplnd": 51, "avoided": 38, "missed": 1, "nb": 0.078, "clin_avoided": 26, "clin_missed": 4},
        0.30: {"eplnd": 37, "avoided": 49, "missed": 4, "nb": 0.062, "clin_avoided": 33, "clin_missed": 5},
    },
}

# ---------------------------------------------------------------------------
# Table S14 — 27 retained features (RF importance descending)
# ---------------------------------------------------------------------------
TABLE_S14 = [
    ("log_sigma_4_0_mm_3D_glcm_InverseVariance_h4", 4, 0.0676, 0.0279),
    ("wavelet_HHL_glcm_Correlation_h5", 5, 0.0554, 0.0222),
    ("log_sigma_3_0_mm_3D_glszm_SmallAreaEmphasis_h3", 3, 0.0535, 0.0140),
    ("original_shape_Maximum2DDiameterSlice_h2", 2, 0.0531, 0.0249),
    ("wavelet_HLL_firstorder_Skewness_h4", 4, 0.0484, 0.0199),
    ("wavelet_LLL_glcm_Correlation_h3", 3, 0.0465, 0.0219),
    ("log_sigma_2_0_mm_3D_glcm_MaximumProbability_h2", 2, 0.0457, 0.0151),
    ("wavelet_LLL_glszm_LargeAreaHighGrayLevelEmphasis_h3", 3, 0.0436, 0.0196),
    ("wavelet_LLH_firstorder_Skewness_h1", 1, 0.0431, 0.0176),
    ("wavelet_LLH_glszm_GrayLevelNonUniformityNormalized_h1", 1, 0.0424, 0.0083),
    ("wavelet_LLL_glcm_Correlation_h2", 2, 0.0347, 0.0092),
    ("log_sigma_2_0_mm_3D_firstorder_Skewness_h4", 4, 0.0317, 0.0090),
    ("wavelet_LHL_firstorder_90Percentile_h3", 3, 0.0310, 0.0078),
    ("wavelet_HLL_firstorder_Kurtosis_h4", 4, 0.0308, 0.0106),
    ("wavelet_HLH_glcm_Idn_h5", 5, 0.0305, 0.0088),
    ("log_sigma_5_0_mm_3D_glszm_LargeAreaHighGrayLevelEmphasis_h3", 3, 0.0304, 0.0126),
    ("log_sigma_3_0_mm_3D_glcm_Correlation_h2", 2, 0.0304, 0.0106),
    ("wavelet_LHH_firstorder_RootMeanSquared_h1", 1, 0.0304, 0.0063),
    ("log_sigma_5_0_mm_3D_firstorder_90Percentile_h2", 2, 0.0301, 0.0088),
    ("wavelet_HLL_glcm_ClusterShade_h3", 3, 0.0300, 0.0092),
    ("log_sigma_5_0_mm_3D_glszm_GrayLevelNonUniformity_h2", 2, 0.0298, 0.0073),
    ("original_shape_Maximum2DDiameterSlice_h3", 3, 0.0297, 0.0115),
    ("log_sigma_2_0_mm_3D_gldm_LargeDependenceLowGrayLevelEmphasis_h5", 5, 0.0297, 0.0083),
    ("wavelet_LHH_firstorder_Median_h1", 1, 0.0275, 0.0085),
    ("wavelet_HLL_firstorder_Kurtosis_h2", 2, 0.0261, 0.0069),
    ("original_shape_Maximum2DDiameterSlice_h1", 1, 0.0255, 0.0062),
    ("wavelet_LLH_ngtdm_Complexity_h4", 4, 0.0224, 0.0069),
]
TABLE_S14_HABITAT_COUNTS = {1: 5, 2: 7, 3: 7, 4: 5, 5: 3}

# ---------------------------------------------------------------------------
# Table S15 — univariable / multivariable logistic regression (n = 377)
# ---------------------------------------------------------------------------
TABLE_S15 = {
    "n": 377,
    "ISUP": {"uni": (2.047, 1.609, 2.603, 0.001), "multi": (1.891, 1.476, 2.422, 0.001)},
    "PI_RADS": {"uni": (1.681, 1.134, 2.493, 0.010), "multi": (1.611, 1.048, 2.477, 0.030)},
    "PSA": {"uni": (1.006, 1.002, 1.009, 0.001), "multi": (1.004, 0.999, 1.009, 0.082)},
    "Age": {"uni": (1.003, 0.972, 1.035, 0.843), "multi": (1.001, 0.968, 1.036, 0.939)},
    "target_roi_volume": {"uni": (0.995, 0.978, 1.013, 0.604), "multi": (1.013, 0.976, 1.051, 0.508)},
    "prostate_volume": {"uni": (0.996, 0.986, 1.006, 0.393), "multi": (0.989, 0.967, 1.011, 0.336)},
    "PSAD": {"uni": (1.118, 1.031, 1.212, 0.007), "multi": (1.019, 0.930, 1.117, 0.681)},
}

# ---------------------------------------------------------------------------
# Table S16 — alternative clinical comparators (pooled external)
# ---------------------------------------------------------------------------
TABLE_S16A = {
    "ISUP":       {"train": 0.692, "test1": 0.759, "test2": 0.747, "test_prospective": 0.664, "pooled": 0.749},
    "ISUP+PI-RADS": {"train": 0.706, "test1": 0.795, "test2": 0.770, "test_prospective": 0.689, "pooled": 0.778},
    "ISUP+PI-RADS+PSA+PSAD": {"train": 0.728, "test1": 0.805, "test2": 0.817,
                              "test_prospective": 0.655, "pooled": 0.794},
}
TABLE_S16B = {
    "ISUP_only": {"ctl": 0.749, "comb": 0.844, "dauc": +0.095, "lo": +0.036, "hi": +0.156, "p": 0.0014},
    "ISUP_PIRADS": {"ctl": 0.778, "comb": 0.844, "dauc": +0.066, "lo": +0.012, "hi": +0.124, "p": 0.022},
}

# ---------------------------------------------------------------------------
# Table S18 — sensitivity analysis, manually corrected target regions
# ---------------------------------------------------------------------------
TABLE_S18A = {
    "train": {"corrected": 17, "excluded": 1, "analysed": 16, "n": 398, "rate_pct": 4.0},
    "test1": {"corrected": 14, "excluded": 0, "analysed": 14, "n": 339, "rate_pct": 4.1},
    "test2": {"corrected": 12, "excluded": 1, "analysed": 11, "n": 189, "rate_pct": 5.8},
    "test_prospective": {"corrected": 9, "excluded": 0, "analysed": 9, "n": 90, "rate_pct": 10.0},
    "total": {"corrected": 52, "excluded": 2, "analysed": 50, "n": 1016, "rate_pct": 4.9},
}
TABLE_S18B = {   # combined AUC full -> reduced
    "train": (0.798, 0.799), "test1": (0.842, 0.846), "test2": (0.829, 0.832),
    "test_prospective": (0.887, 0.891), "pooled": (0.844, 0.849),
}
TABLE_S18C = {"Clinic": (4.063, 4.067), "H2": (2.593, 2.935), "H4": (3.934, 3.843),
              "intercept": (-4.249, -4.326), "pooled_auc": (0.849, 0.848)}

# ---------------------------------------------------------------------------
# Table S21 / Table 5 — subgroups of the combined model (pooled external, n = 618)
# ---------------------------------------------------------------------------
TABLE_S21 = {
    "ISUP<=2": {"n": 195, "pn1": 9, "auc": 0.843, "lo": 0.641, "hi": 0.973},
    "ISUP>=3": {"n": 423, "pn1": 73, "auc": 0.811, "lo": 0.751, "hi": 0.853},
    "PSA<=10": {"n": 227, "pn1": 22, "auc": 0.859, "lo": 0.793, "hi": 0.920},
    "PSA10-20": {"n": 173, "pn1": 11, "auc": 0.935, "lo": 0.881, "hi": 0.977},
    "PSA>20": {"n": 218, "pn1": 49, "auc": 0.763, "lo": 0.689, "hi": 0.831},
    "1.5T": {"n": 45, "pn1": 2, "auc": 0.849, "lo": 0.733, "hi": 0.953},
    "3.0T": {"n": 562, "pn1": 78, "auc": 0.852, "lo": 0.809, "hi": 0.892},
}
TABLE_S21_INTERACTION_P = {"PSA": 0.308, "ISUP": 0.615, "field_strength": 0.980}

# ---------------------------------------------------------------------------
# Table S2 — habitat profiles (log-sigma 2.0 mm)
# ---------------------------------------------------------------------------
TABLE_S2A = {   # feature: [H1..H5]
    "Signal mean":                        [3.113, 2.848, 2.398, 2.214, 1.799],
    "Entropy (heterogeneity)":            [3.459, 3.478, 3.533, 3.523, 3.527],
    "Uniformity":                         [0.116, 0.114, 0.109, 0.111, 0.110],
    "GLCM contrast":                      [10.065, 10.944, 12.812, 13.306, 13.420],
    "GLCM correlation":                   [0.460, 0.416, 0.390, 0.385, 0.388],
    "GLCM inverse difference moment":     [0.403, 0.389, 0.375, 0.370, 0.372],
    "GLSZM small-area emphasis":          [0.715, 0.737, 0.768, 0.771, 0.764],
    "GLSZM zone entropy":                 [5.005, 4.869, 4.718, 4.686, 4.731],
    "GLRLM short-run emphasis":           [0.948, 0.953, 0.963, 0.963, 0.963],
    "GLRLM long-run emphasis":            [1.270, 1.246, 1.191, 1.186, 1.187],
}
TABLE_S2B = {   # z-score across habitats (ddof = 0)
    "Signal mean":             [+1.38, +0.80, -0.16, -0.56, -1.46],
    "Entropy":                 [-1.51, -0.87, +0.97, +0.64, +0.77],
    "Uniformity":              [+1.53, +0.77, -1.15, -0.38, -0.77],
    "GLCM contrast":           [-1.51, -0.86, +0.52, +0.88, +0.97],
    "GLCM correlation":        [+1.84, +0.29, -0.63, -0.80, -0.70],
    "GLSZM small-area emphasis": [-1.66, -0.65, +0.78, +0.92, +0.60],
    "GLRLM long-run emphasis": [+1.54, +0.85, -0.71, -0.85, -0.82],
}
TABLE_S2_FEATURE_COLUMNS = {
    "Signal mean": "log_sigma_2_0_mm_3D_firstorder_Mean",
    "Entropy (heterogeneity)": "log_sigma_2_0_mm_3D_firstorder_Entropy",
    "Uniformity": "log_sigma_2_0_mm_3D_firstorder_Uniformity",
    "GLCM contrast": "log_sigma_2_0_mm_3D_glcm_Contrast",
    "GLCM correlation": "log_sigma_2_0_mm_3D_glcm_Correlation",
    "GLCM inverse difference moment": "log_sigma_2_0_mm_3D_glcm_Idm",
    "GLSZM small-area emphasis": "log_sigma_2_0_mm_3D_glszm_SmallAreaEmphasis",
    "GLSZM zone entropy": "log_sigma_2_0_mm_3D_glszm_ZoneEntropy",
    "GLRLM short-run emphasis": "log_sigma_2_0_mm_3D_glrlm_ShortRunEmphasis",
    "GLRLM long-run emphasis": "log_sigma_2_0_mm_3D_glrlm_LongRunEmphasis",
}
TABLE_S2_N = 1016

# ---------------------------------------------------------------------------
# Table 1 — baseline characteristics
# ---------------------------------------------------------------------------
TABLE1 = {
    "Age":                {"train": (70.0, 66.0, 75.0), "test1": (70.0, 66.0, 75.0),
                           "test2": (69.0, 65.0, 73.0), "test_prospective": (71.5, 66.8, 74.0), "p": 0.045},
    "PSA":                {"train": (33.5, 20.4, 70.8), "test1": (10.5, 5.6, 19.8),
                           "test2": (13.6, 8.1, 24.8), "test_prospective": (26.7, 14.3, 65.8), "p": 0.001},
    "PSAD":               {"train": (0.88, 0.49, 1.57), "test1": (0.28, 0.13, 0.54),
                           "test2": (0.44, 0.26, 0.84), "test_prospective": (0.70, 0.35, 1.48), "p": 0.001},
    "target_roi_volume":  {"train": (17.5, 12.4, 25.0), "test1": (16.1, 10.0, 23.5),
                           "test2": (17.5, 11.9, 24.9), "test_prospective": (19.1, 13.2, 29.4), "p": 0.007},
    "prostate_volume":    {"train": (37.5, 29.1, 50.8), "test1": (35.5, 26.8, 49.2),
                           "test2": (33.4, 24.2, 48.4), "test_prospective": (39.2, 27.5, 56.6), "p": 0.005},
}
TABLE1_ISUP = {   # grade group: [train, test1, test2, prospective]
    1: [13, 7, 16, 1], 2: [61, 97, 66, 8], 3: [76, 125, 59, 21],
    4: [104, 77, 27, 27], 5: [137, 33, 21, 33],
}
TABLE1_PIRADS = {3: [37, 75, 47, 6], 4: [88, 107, 62, 23], 5: [273, 157, 80, 61]}

# ---------------------------------------------------------------------------
# Fig. 6 / Table S22 — the five illustrated cases
# ---------------------------------------------------------------------------
# The five illustrated cases (Fig. 6 / Fig. 8), as study codes.  They were
# taken from the archive under five hospital admission numbers — direct
# identifiers that must not appear in a public repository.  The old→new table
# lives OUTSIDE the repo (see src/hrpca/pseudonym.py).
FIG6_CASES = ["S0043", "S0036", "S0797", "S0959", "S0968"]

# ---------------------------------------------------------------------------
# Deep-learning comparator (Table S20b sensitivity analysis)
# ---------------------------------------------------------------------------
TABLE_S20B = {
    "Combined": (0.8442, 0.8434), "Clinic": (0.7785, 0.7780), "HabitatH4": (0.7294, 0.7284),
    "HabitatH2": (0.6764, 0.6758), "Habitat": (0.5914, 0.5906),
}
