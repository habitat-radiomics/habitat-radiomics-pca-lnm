# results/ — generated output

**This directory is empty on a fresh clone.**  Everything here is produced by
running the pipeline; nothing is committed.

| Sub-directory | Written by | Content |
|---|---|---|
| `results/tables/` | `src/05`–`src/09`, `verify/` | recomputed tables and verification reports |
| `results/figures/` | `src/10_figures.py` | Fig. 2, 3a, 3b, S2, S7, S9 (`.png` 600 dpi + `.pdf`) |
| `results/logs/` | `run_all.py` | run logs |

## With the real analysis dataset

Requires the individual-level tables, which are **not** public
(ChiCTR2500111002: *IPD sharing: No*) — obtainable from the corresponding author
under a data-use agreement and with institutional approval.  Once `data/` is
populated:

```bash
python -X utf8 run_all.py
```

This reproduces every table and figure and runs the four numeric verifiers, each
of which must report **FAIL 0**.

## Without it

The numeric verifiers exit **3** ("the individual-level analysis dataset is not
present") rather than passing on an empty dataset.  To exercise the code path
instead:

```bash
python tools/make_synthetic_data.py
python verify/check_paper_numbers.py --synthetic     # pipeline demonstration only
```

The synthetic run proves the code executes; its numbers are synthetic and are
**not** compared with the manuscript.

> Earlier revisions of this repository shipped `results/` generated from the real
> dataset.  Those outputs were moved to the controlled institutional store
> together with the per-patient tables, because they are derived from
> individual-level data.
