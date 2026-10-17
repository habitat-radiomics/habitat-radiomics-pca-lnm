# The `real_id` training-cohort row shift

**Status: fixed.**  Found 2026-10-17, during a review query about the deep-learning
comparator.  Affects only analyses that traversed the `real_id` bridge; **no
published number is affected.**

---

## 1. Summary

`scores/*.csv` carry no identifier — only a within-cohort `patient_index`.  The
join to patient-level data therefore went through `cases_with_ids.csv`, whose
`real_id` column is the only bridge between a score row and a study code.

In the **training cohort** that bridge was shifted by one row: the identifier on
score row *i* belonged to row *i+1*.  The two external cohorts were correct; the
prospective cohort was shifted in 7 of 90 rows.

| cohort | rows | `real_id` correct, before | after |
|---|---|---|---|
| train | 398 | **43 / 398** | **398 / 398** |
| test1 | 339 | 339 / 339 | 339 / 339 |
| test2 | 189 | 189 / 189 | 189 / 189 |
| test_prospective | 90 | **83 / 90** | **90 / 90** |

*(7-column key; see §3.  Restricting to the two-column `(H2, H4)` fingerprint,
which can only resolve rows whose rounded H₂/H₄ pair is unique, the training
cohort went from 34/340 to 340/340.)*

Aggregate effect: the score-file label agreed with the archive label for
**874 / 1,016** rows before, **1,016 / 1,016** after.

## 2. Why it mattered

The training scores were therefore evaluated against the neighbouring patient's
outcome.  The deployed combined model — which is genuinely good — appeared
broken:

| quantity | through the broken `real_id` | correct connection | published |
|---|---|---|---|
| `Combined`, training AUC | **0.570** | **0.798** | 0.798 |
| `Combined`, pooled external | 0.834 | **0.844** | 0.844 |
| `Clinical`, training AUC | 0.580 | **0.706** | 0.704 |
| `all-habitat`, training AUC | 0.555 | **0.864** | 0.864 |

The external cohorts were already correct, which is why only the *training* AUC
collapsed and the pooled AUC moved by only 0.01 — the signature that pointed at
the join rather than at the model.

## 3. How it was found

Two independent lines of evidence, neither of which needed the identifier.

**(a) Numeric fingerprint.**  `scores_highprec.csv` and `fullprec_scores.csv`
carry the same probabilities **with** the study code, at 16 significant digits.
The deployed `scores/*.csv` are the same values at reduced precision.  Rounding a
row's seven signature probabilities to the two decimals at which the deployed
files were released gives a key that is **unique for all 1,016 patients** — so
every deployed score row can be resolved to a patient without touching
`real_id`.  Against that resolution, `real_id` was correct for only 43/398
training rows.

*Precision note.*  A two-column `(H2, H4)` fingerprint is **not** sufficient for
the training cohort: `scores/train.csv` stores two decimals, so 398 rows collapse
to only 60 distinct H₂ values and 58 rows remain ambiguous.  The 7-column vector
resolves them all.

**(b) Clinical monotonicity.**  The probability of pN1 must rise with the ISUP
grade group.  Scored against the labels reached through the broken `real_id` it
did not; against the fingerprint-resolved pairing it did:

| ISUP | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| pN1 rate, broken join | 0.0 % | 11.5 % | 24.1 % | 22.1 % | **49.6 %** |
| pN1 rate, correct join | 7.7 % | 31.1 % | 26.5 % | 34.6 % | 29.2 % |

The broken column is the one that is *supposed* to be monotone, and the correct
one is not — that inversion is what made the error visible.

Neither check uses `real_id`, so neither can inherit its defect.

## 4. Root cause and fix

The defect is in the **archived source file** `scripts/_cases_with_ids.csv`
itself, not in the copy step: the archive file shows the same 34/340 training
agreement as the copy in `data/`.

`src/00_assemble_data.py` previously copied that file verbatim.  It now calls
`_repair_real_id()` immediately after the copy, which

1. builds a per-cohort key from the **seven** signature probabilities rounded to
   the released precision, taken from `scores_highprec.csv` (which carries the
   study code);
2. resolves every row of `cases_with_ids.csv` against that key;
3. **raises** if any row fails to resolve or if two patients share a key — it
   never guesses and never falls back silently;
4. cross-checks the rebuilt identifier two ways: it must exist in
   `model_inputs.csv`, and it must be the patient whose `(H2, H4)` pair the row
   rounds to;
5. rewrites `cases_with_ids.csv` and updates its `MANIFEST.json` entry.

The repair is idempotent and deterministic: running it on the archived file
changes 362 of 1,016 rows, and running it again changes nothing.

## 5. Scope of impact

| Affected | Not affected |
|---|---|
| the supplementary `Clinical + all-habitat` models that read `H2`/`H4` off the score files, and the `B (H2 + H4 only)` model in particular | every published AUC, CI, calibration statistic and decision-curve value — those were computed **within** each score file, where the label column is self-consistent |
| any analysis that joined score rows to `Clinic.csv` / `label.csv` / `model_inputs.csv` through `real_id` | `A1`/`A2`/`A3`, which use the 27 habitat features and the two clinical variables, all keyed by study code and never touching `real_id` |
| the per-patient prediction export produced before the fix | the `scores/test1`, `scores/test2` joins (already correct) |

**Published numbers are unaffected** because the publication-side pipeline never
needed the bridge: each score file already contains its own `label` column, and
that column was correct.  The defect only bit analyses that *cross* files.

## 6. How to avoid a recurrence

* Never join the score files on row position or on `patient_index` alone.
* Prefer the keyed exports (`scores_highprec.csv`, `fullprec_scores.csv`), which
  carry the study code next to the value.
* When a keyed export is unavailable, resolve by a **multi-column** numeric
  fingerprint and require uniqueness; a two-column fingerprint is too weak once
  values are rounded.
* Sanity-check any new join with a monotonicity property that does not depend on
  the join being right (here: pN1 rate versus ISUP grade).

## 7. Related change in the synthetic set

`tools/make_synthetic_data.py` used to emit `real_id = SYN0001…SYN0010` **within
each cohort**, so the same identifier recurred in all four cohorts and a join on
`real_id` alone was ambiguous — the same class of defect, in miniature.  It now
emits `SYN-TRN0-0001`, `SYN-TST1-0001`, …, which is unique across cohorts.  The
synthetic set is regenerated and its column names and order still match the real
tables exactly; the demonstration in `verify/check_paper_numbers.py --synthetic`
passes.
