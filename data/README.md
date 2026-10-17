# Data availability

> **The individual-level analysis dataset is not publicly available; it may be
> obtained from the corresponding author under a data-use agreement and with
> institutional approval.  A synthetic dataset with identical structure is
> provided so the pipeline can be run end-to-end.**

The trial registration (ChiCTR2500111002) states *IPD sharing: No*.  Accordingly
this directory contains **no per-patient record**.  It holds the aggregate
results, the metadata, and a **fabricated** dataset that mirrors the real schema.

## What ships

| Path | Rows | Content | Level |
|---|---|---|---|
| `synthetic/clinical.csv` | 40 | `ID, Age, PI_RADS, PSA, ISUP, target_roi_volume, prostate_volume, PSAD` | ★ **synthetic** |
| `synthetic/labels.csv` | 40 | `ID, label, group` | ★ synthetic |
| `synthetic/analysis_set.csv` | 40 | `ID, group, label` | ★ synthetic |
| `synthetic/model_inputs.csv` | 40 | `ID, group, label, ISUP, PI_RADS, H2, H4` | ★ synthetic |
| `synthetic/cases_with_ids.csv` | 40 | the identifier-key layout | ★ synthetic |
| `synthetic/scores/*.csv` | 40 total | per-signature probability layout | ★ synthetic |
| `synthetic/features/rad_features_h{1..5}.csv` | 40 × 1,198 | habitat feature layout (`--features`) | ★ synthetic |
| `K_CV10x10_summary.csv` | 8 | 100-fold repeated-CV summary by K | aggregate |
| `dl_comparator_results.csv`, `dl_bootstrap_ci.csv` | — | 3D ResNet-18 comparator, pooled | aggregate |
| `MANIFEST.json` | — | SHA-256 of every assembled file | metadata |

> **Synthetic data for pipeline demonstration only.**  Every value under
> `synthetic/` comes from a seeded random number generator
> (`tools/make_synthetic_data.py`, seed 20260101).  The real tables were
> consulted **only** to copy column names; no real cell value is present.
> Numbers computed on these files are meaningless and will not match the paper.

## What does **not** ship, and why

| Data | Rows | Reason |
|---|---|---|
| `clinical.csv`, `clinical_workspace.csv` | 1,016 / 1,097 | individual-level clinical variables — **IPD, not shareable** |
| `labels.csv`, `analysis_set.csv`, `model_inputs.csv` | 1,125 / 1,016 / 1,016 | per-patient reference standard and cohort assignment — IPD |
| `scores/*.csv`, `scores_highprec.csv`, `fullprec_scores.csv` | 398 / 339 / 189 / 90 | **per-patient predicted probabilities** — IPD |
| `k_runs/K{4,5,7}.csv` | 1,016 / 1,016 / 993 | per-patient scores of the K-specific runs — IPD |
| `features/rad_features_h*.csv`, `features/k{4,7}_*` | 1,025 × 1,198 each | **per-patient radiomics features** — IPD |
| `shap/*` | 1,009 + 27 | per-patient SHAP inputs and values — IPD |
| `cases_with_ids.csv` | 1,016 | the **identifier key** — direct identifiers |
| `results/` run outputs | — | derived from the per-patient tables |
| DICOM / NIfTI images, masks, nnU-Net and ResNet weights | — | patient privacy and IRB terms |
| WTR comparator per-patient scores | — | produced by a separate archived run |

The controlled copies live in an institutional store outside version control and
are released only under a data-use agreement with institutional approval.

## Reproducing

```bash
python tools/make_synthetic_data.py                  # rebuild the synthetic set
python verify/check_paper_numbers.py --synthetic     # pipeline demonstration
```

With the controlled-access tables placed in this directory, the four numeric
verifiers run against the real data instead (see the top-level README §4/§8).

## Ethics

This retrospective multicentre study followed the Declaration of Helsinki and was
approved by the institutional review board of the First Affiliated Hospital of
Anhui Medical University (MR-34-25-056535); data from the collaborating centres
were provided under their respective institutional data-sharing approvals.
Written informed consent was waived for the retrospective cohorts.
