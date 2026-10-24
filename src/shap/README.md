# SHAP figure code

Reproduces the SHAP figures of the paper:

* **Fig. 5** — per-model beeswarm and mean |SHAP| panels for the H2 (7 features)
  and H4 (18 features) habitat models → `h2_h4_shap.py`
* **Fig. 6 / Fig. S10** — per-case three-level SHAP decomposition, force plots and
  the five-case stack → `shap_FINAL_v3.py`, `draw_5cases_native.py`,
  `combo_shap_mri.py`, `stack_5cases.py`

The SHAP values and force plots can be reproduced from the feature table and the
published coefficients; the MRI panels in Fig. 6 require the source images,
which are not distributed.

## Install

```bash
python -m pip install -r requirements.txt
```

Figures use **Arial**, with **Microsoft YaHei** as the fallback for the circled
digits ①②③ (Arial has no U+2460 block). Both are requested through
`plt.rcParams["font.family"]`; no font path is hard-coded.

## Environment variables

No absolute path is hard-coded. Set the ones each script needs; a missing
variable stops the script with a message naming it.

| Variable | Meaning |
|---|---|
| `HRPCA_SHAP_FEATURES` | pooled `rad_features.csv` (must be the version that matches the project model — **not** the older `rad_features_h2.csv` / `rad_features_h4.csv`, whose values differ) |
| `HRPCA_SHAP_LABELS` | `label.csv`, columns `ID, group, label` |
| `HRPCA_SHAP_CLINIC` | `Clinic.csv`, columns `ID, Age, PI-RADS, PSA, ISUP, volume_tumor, volume_prostate, PSAD` |
| `HRPCA_SHAP_MODELS` | directory holding `HabitatH2__LR_label.pkl` and `HabitatH4__LR_label.pkl`. These pkl files store the **feature-name lists** (7 and 18 names), not fitted estimators |
| `HRPCA_SHAP_RESULTS` | the project `results/` directory with `HabitatH2_LR_{train,test}.csv`, `HabitatH4_LR_*.csv`, `Clinic_LR_*.csv` (column `label-1` = predicted probability) |
| `HRPCA_SHAP_OUT` | output root; every script writes a fixed subdirectory inside it |
| `HRPCA_SHAP_CASES` | optional, `combo_shap_mri.py` only: directory with `FigS10_{a-e}_{id}.png`; defaults to `$HRPCA_SHAP_OUT/cases` |
| `HRPCA_SHAP_MRI` | `combo_shap_mri.py` only: directory with the MRI panel SVGs, named `{case}_{habitat}.svg` (e.g. `case_a_h4.svg`). **Not distributed** |
| `HRPCA_SHAP_EXPORT` | optional, `draw_5cases_native.py` only: extra directory to copy the force plots into |

`PANEL_ORDER=mri_top` flips the composite panels so the MRI row sits on top.

## Scripts and run order

### 1 · `h2_h4_shap.py` — Fig. 5

StandardScaler + LogisticRegression refitted on the **training cohort** (n = 398)
for each habitat model's own feature list, then `shap.LinearExplainer`. Writes
`$HRPCA_SHAP_OUT/h2_h4_shap/`:

```
Fig_SHAP_H2_beeswarm.png/.svg   Fig_SHAP_H2_bar.png/.svg
Fig_SHAP_H4_beeswarm.png/.svg   Fig_SHAP_H4_bar.png/.svg
SHAP_H2_ranking.csv  SHAP_H4_ranking.csv
SHAP_H2_values_train.csv  SHAP_H4_values_train.csv  summary_H2_H4.csv
```

### 2 · `shap_FINAL_v3.py` — the 26-item decomposition (**current**)

The combined model, on the probability scale:

```
L = -4.248788 + 4.062609·p_Clinic + 2.593122·p_H2 + 3.934389·p_H4
BASE = -4.248788 + 4.062609·0.297739 + 2.593122·0.296332 + 3.934389·0.296558
     = -1.1039 log-odds   (p = 0.2490)
```

The component probabilities are the project's own predictions, so the case values
match the paper digit for digit. The decomposition assigns each component's
contribution `w·Δp` to its features in proportion to `β_k(z_k − E_k) / Σβ(z − E)`:

| block | items |
|---|---|
| H4 features | 18 |
| H2 features | 7 |
| `Clinic probability` | **1** |
| **total** | **26** |

Closure is asserted: `max |BASE + Σ SHAP − L| < 1e-9`. Writes
`$HRPCA_SHAP_OUT/h2h4_final/` (`SH_all.npy` 1009×26, `SH_feature_names.csv`,
`meta_all.csv`, `base.csv`).

### 3 · `shap_FINAL_v2_alt_clinic_split.py` — the 27-item decomposition

Same model and same base value, but the clinic block is expanded into its two
standardised inputs, `Clinical: ISUP` and `Clinical: PI-RADS`:

| block | items |
|---|---|
| H4 features | 18 |
| H2 features | 7 |
| `Clinical: ISUP` + `Clinical: PI-RADS` | **2** |
| **total** | **27** |

**This is the form assumed by the figure labels.** The key in Supplemental Note S1
lists `ClinISUP` / `ClinPIRADS` as separate labels, and
`TableS22_Fig6_label_identity.csv` describes its residual row as "pooled
contribution of the remaining **19** features" (8 shown + 19 = 27). The two
scripts therefore disagree by one item, and the delivered Fig. 6 follows this one.
It writes to its own directory, `$HRPCA_SHAP_OUT/h2h4_final_v2_clinic_split/`, so
running it does not overwrite the v3 output (in the archive both scripts wrote to
the same folder, so whichever ran last silently won).

### 4 · `draw_5cases_native.py` — force plots

`shap.force_plot(..., matplotlib=True)`, top-8 features plus an `N others` segment
that still enters f(x); labels `①LoG…h4=+4.49` (index + short name + habitat +
z-score). Works with either the 26- or the 27-item matrix. Writes
`$HRPCA_SHAP_OUT/native/Fig_force_fusion_{a-e}_{id}.png/.svg`.

### 5 · `combo_shap_mri.py` — SHAP + MRI composite

Decodes the base64 PNGs embedded in the supplied MRI SVGs, rescales the row to
the SHAP figure width, moves the panel labels underneath the panels and stacks the
two rows. Writes `$HRPCA_SHAP_OUT/FigS10_combo/`.

### 6 · `stack_5cases.py` — the five-case figure

`$HRPCA_SHAP_OUT/FigS10_all_cases.png` (4412 × 5462) and a 2600 px copy.

### 7 · `find_clinic_model_vars.py` — supporting check

Enumerates all 127 subsets of the clinical variables and matches each against the
project's `Clinic_LR_*.csv`. The answer for the archived run is **PI-RADS + ISUP**
(r = 0.9998, mean absolute difference 0.0030) — the justification for the two
clinical items in the 27-item form.

## Reproduction check

| case | paper p(LNM) | this code | Δ |
|---|---|---|---|
| case_a | 0.7889 | 0.7889 | 0.0000 |
| case_b | 0.0358 | 0.0358 | 0.0000 |
| case_c | 0.6144 | 0.6144 | 0.0000 |
| case_d | 0.0510 | 0.0510 | 0.0000 |
| case_e | 0.0508 | 0.0508 | 0.0000 |

Both decomposition scripts assert the closure `BASE + Σ SHAP = L` to < 1e-9.

## Visual conventions

| item | setting |
|---|---|
| font | Arial, circled digits fall back to Microsoft YaHei |
| colours | push `#E8174B` (red), pull `#2E7BCF` (blue), `plot_cmap="RdBu"` |
| output | 300 dpi PNG + vector SVG |
| labels | `①short…hN=value`, underscores rendered as `…` |
| panel labels | underneath the MRI panels |

## Notes

1. `models/*.pkl` hold **feature-name lists**, not fitted models. The per-habitat
   models are refitted here on the training cohort; the refit reproduces the
   project's official predictions at r = 1.00000.
2. `shap.force_plot` only draws segments wide enough to label. A feature whose
   |SHAP| is very small in a given case (PI-RADS in a low-grade case, say) is
   folded into `others`, but it **still enters f(x)**.
3. The MRI panels are not distributed, so steps 5–6 can only be run by a reader
   with the imaging data.
