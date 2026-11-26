# habitat-radiomics-pca-lnm

Analysis code and aggregate result tables for

> **Habitat selection by clinical endpoint predicts nodal status in prostate cancer**

## Contents

| Path | What it is |
|---|---|
| src/ | Analysis pipeline: segmentation, habitat clustering, feature selection, modelling, figures |
| supplementary_data/ | 20 aggregate result tables (five are also supplied as Supplementary Data 01-05) |
| results/ | Generated outputs (not committed; see results/README.md) |
| verify/ | Numeric verification scripts |
| docs/ | Verification notes, provenance, configuration |

## Quick start

    pip install -r requirements.txt
    python -X utf8 verify/check_paper_numbers.py --synthetic

## Data availability

Individual-level data are not publicly available (ChiCTR2500111002, IPD sharing: No);
they are available from the corresponding author under a data-use agreement. Aggregate
results underlying the reported analyses are provided in supplementary_data/ and in the
Supplementary Information. No imaging data are included.

## Ethics

Approved by the Clinical Research Ethics Committee of the First Affiliated Hospital of Anhui
Medical University (PJ 2025-04-44; MR-34-25-056535), which waived written informed consent for
the retrospective cohorts; participants in the prospective cohort gave written consent.

## Environment

Python 3.9+. No GPU required.

## Citation and licence

See CITATION.cff and LICENSE.

Detailed verification notes: docs/VERIFICATION_NOTES.md
