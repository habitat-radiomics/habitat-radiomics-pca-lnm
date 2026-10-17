#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the complete from-derived-data pipeline, then verify every published number.

    python run_all.py                     # statistics + figures + verification
    python run_all.py --with-assemble     # also rebuild data/ from the archive first
    python run_all.py --verify-only       # only the verification step

Steps that need the imaging data (segmentation, clustering, feature extraction)
are reported as skipped with the reason, rather than silently omitted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (script, extra args, needs imaging data)
STEPS = [
    ("src/00_assemble_data.py", ["--skip-features"], False, "assemble data/ from the archive"),
    ("src/02_habitat_clustering.py", ["--report"], False, "Table S3 / Fig. S2 — K selection"),
    ("src/03_feature_extraction.py", ["--profiles"], False, "Table S2 — habitat profiles"),
    ("src/05_model_training.py", [], False, "Table 2, Table 3, combined coefficients"),
    ("src/06_validation.py", [], False, "Table 2 CIs, Table 4, Table S12"),
    ("src/07_calibration.py", [], False, "Table S9, Table S17a, Brier"),
    ("src/08_dca.py", [], False, "Table S13, Table S17b/c, Fig. S4 data"),
    ("src/09_shap.py", [], False, "Fig. 4 / Fig. S9 / Fig. 6"),
    ("src/10_figures.py", [], False, "all figures"),
]
IMAGING_STEPS = [
    ("src/01_segmentation.py", ["--check", "--masks", "<masks>"], "nnU-Net segmentation QC"),
    ("src/02_habitat_clustering.py", ["--criteria", "<voxels.npy>"], "Table S3c internal criteria"),
    ("src/03_feature_extraction.py", ["--candidates"], "1,197-column candidate matrix"),
    ("src/04_feature_selection.py", ["--matrix", "<matrix.csv>", "--lasso"], "feature selection"),
]


def run(cmd: list, label: str) -> bool:
    print(f"\n{'=' * 78}\n  {label}\n  $ {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"  -> {'OK' if ok else 'FAILED'} ({dt:.1f} s)", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-assemble", action="store_true",
                    help="rebuild data/ from the archive (reads absolute paths in config.yaml)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--include-features", action="store_true",
                    help="with --with-assemble, also copy the ~115 MB feature tables")
    ap.add_argument("--config", default=None,
                    help="alternative config; its data_dir may point at the "
                         "controlled store")
    args = ap.parse_args()
    cfgf = ["--config", args.config] if args.config else []

    failures = []
    if not args.verify_only:
        steps = STEPS if args.with_assemble else STEPS[1:]
        for script, extra, _, label in steps:
            if script == "src/00_assemble_data.py" and not args.include_features:
                extra = ["--skip-features"]
            if not run([PY, "-X", "utf8", script, *extra], label):
                failures.append(script)

    print(f"\n{'=' * 78}\n  ★ verification — paper value vs code value\n{'=' * 78}", flush=True)
    proc = subprocess.run([PY, "-X", "utf8", "verify/check_paper_numbers.py", "-v", *cfgf],
                          cwd=str(ROOT))
    if proc.returncode == 3:
        print("\n  → SKIPPED: the submitted supplement is incomplete (mid-rebuild);"
              "\n    this is an environment problem, not a numerical failure.")
    elif proc.returncode != 0:
        failures.append("verify/check_paper_numbers.py")

    # both published-value verifiers parse the submitted documents at run time,
    # so exit 3 = the submission workspace is mid-rebuild (environment, not data)
    for script, label in [("verify/check_disputed_items.py", "disputed items")]:
        proc = subprocess.run([PY, "-X", "utf8", *script.split(), *cfgf], cwd=str(ROOT))
        if proc.returncode == 3:
            print(f"\n  → SKIPPED ({label}): the submitted supplement is incomplete"
                  " (mid-rebuild); environment problem, not a numerical failure.")
        elif proc.returncode != 0:
            failures.append(script)

    for script, label in [("verify/check_id_linkage.py", "identifier key"),
                          ("verify/check_figure_values.py",
                           "figure ↔ table numeric agreement"),
                          ("verify/scan_identifiers.py --gate",
                           "identifier gate (no admission numbers may ship)")]:
        cmd = [PY, "-X", "utf8", *script.split(), *cfgf]
        if not run(cmd, label):
            failures.append(script)

    print(f"\n{'=' * 78}\n  steps that need the imaging data (NOT distributed)\n{'=' * 78}")
    for script, extra, label in IMAGING_STEPS:
        print(f"  {label:<42} python {script} {' '.join(extra)}")

    print(f"\n{'=' * 78}")
    if failures:
        print(f"  FAILED: {failures}")
    else:
        print("  everything that can run from the derived data completed successfully")
    print(f"{'=' * 78}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
