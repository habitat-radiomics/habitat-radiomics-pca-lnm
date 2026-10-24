#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01 — gland segmentation (3D nnU-Net).

The whole prostate gland is segmented on axial T2WI with a 3D U-Net trained
with nnU-Net.  The network and the raw DICOM/NIfTI data are **not** part of
this repository: imaging data cannot be shared (patient privacy and IRB
restrictions), and the trained weights are tied to the institutional data.

This module therefore does three things:

1. documents the exact inference configuration used in the study
   (``config/nnunet_*`` are the nnU-Net plans/predict settings);
2. provides a thin, reproducible wrapper around ``nnUNetv2_predict`` so that a
   reader with data access can regenerate the masks with the same settings;
3. if a directory of pre-computed masks is supplied, verifies the masks
   (geometry, non-empty, binary) so that downstream steps start from a checked
   input.

Segmentation quality in the study: mean Dice 0.886 on 13 held-out cases
(nnU-Net EMA pseudo-Dice at internal validation).

Usage
-----
    # verify an existing mask directory
    python src/01_segmentation.py --masks /path/to/nnunet/predictions --check

    # run nnU-Net inference (requires the trained weights and raw data)
    python src/01_segmentation.py --images /path/to/images --out /path/to/masks --run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrpca import banner, ensure_dir, load_config, log, paths, write_json  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen inference configuration
# ---------------------------------------------------------------------------
NNUNET_CONFIG = {
    "trainer": "nnUNetTrainer_250epochs",
    "plans": "nnUNetPlans",
    "configuration": "3d_fullres",
    "folds": [0, 1, 2, 3, 4],
    "checkpoint": "checkpoint_final.pth",
    "input_sequence": "T2WI (axial, co-registered target grid)",
    "target_spacing_mm": [1.0, 1.0, 1.0],
    "label": {"0": "background", "1": "whole prostate gland"},
    "preprocessing": ["N4 bias-field correction", "resample to 1x1x1 mm^3",
                      "z-score intensity normalisation"],
    "reported_quality": {"metric": "Dice", "value": 0.886, "n_cases": 13,
                         "reference": "held-out internal validation cases",
                         "note": "nnU-Net EMA pseudo-Dice"},
}


def check_masks(mask_dir: Path) -> dict:
    """Verify geometry and content of a directory of predicted masks."""
    import numpy as np
    import SimpleITK as sitk

    files = sorted(mask_dir.glob("*.nii.gz")) + sorted(mask_dir.glob("*.nii"))
    log(f"  {len(files)} mask files in {mask_dir}")
    report = {"n_files": len(files), "bad": [], "spacings": {}, "volumes_ml": []}
    for f in files:
        try:
            img = sitk.ReadImage(str(f))
            arr = sitk.GetArrayFromImage(img)
        except Exception as exc:                      # pragma: no cover
            report["bad"].append({"file": f.name, "error": str(exc)})
            continue
        labels = np.unique(arr)
        spacing = tuple(round(float(s), 4) for s in img.GetSpacing())
        report["spacings"][str(spacing)] = report["spacings"].get(str(spacing), 0) + 1
        if not set(labels.tolist()) <= {0, 1}:
            report["bad"].append({"file": f.name, "error": f"non-binary labels {labels.tolist()}"})
            continue
        vox = float((arr == 1).sum())
        if vox == 0:
            report["bad"].append({"file": f.name, "error": "empty mask"})
            continue
        report["volumes_ml"].append(vox * float(np.prod(img.GetSpacing())) / 1000.0)
    if report["volumes_ml"]:
        v = np.asarray(report["volumes_ml"])
        report["volume_ml"] = {"min": float(v.min()), "median": float(np.median(v)),
                               "max": float(v.max())}
    log(f"  spacings: {report['spacings']}")
    if "volume_ml" in report:
        log(f"  gland volume (mL): min {report['volume_ml']['min']:.1f}  "
            f"median {report['volume_ml']['median']:.1f}  max {report['volume_ml']['max']:.1f}")
    log(f"  problems: {len(report['bad'])}")
    for b in report["bad"][:10]:
        log(f"    - {b['file']}: {b['error']}")
    return report


def run_nnunet(images: Path, out: Path, cfg: dict, nnunet_root: str | None) -> None:
    """Invoke ``nnUNetv2_predict`` with the frozen configuration."""
    ensure_dir(out)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(images),
        "-o", str(out),
        "-d", str(cfg.get("dataset_id", "001")),
        "-c", NNUNET_CONFIG["configuration"],
        "-tr", NNUNET_CONFIG["trainer"],
        "-p", NNUNET_CONFIG["plans"],
        "-f", *[str(f) for f in NNUNET_CONFIG["folds"]],
        "-chk", NNUNET_CONFIG["checkpoint"],
    ]
    log("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--masks", default=None, help="directory of pre-computed masks")
    ap.add_argument("--images", default=None, help="directory of raw images (for --run)")
    ap.add_argument("--out", default=None, help="output directory (for --run)")
    ap.add_argument("--check", action="store_true", help="verify a mask directory")
    ap.add_argument("--run", action="store_true", help="call nnUNetv2_predict")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    banner("01 · gland segmentation (3D nnU-Net)")
    log(json.dumps(NNUNET_CONFIG, indent=2, ensure_ascii=False))

    out_dir = P["results"] / "segmentation"
    ensure_dir(out_dir)
    write_json(NNUNET_CONFIG, out_dir / "nnunet_config.json")

    if args.check:
        if not args.masks:
            log("  --check needs --masks")
            return 2
        rep = check_masks(Path(args.masks))
        write_json(rep, out_dir / "mask_qc.json")
        return 1 if rep["bad"] else 0

    if args.run:
        if not (args.images and args.out):
            log("  --run needs --images and --out")
            return 2
        run_nnunet(Path(args.images), Path(args.out), cfg.get("nnunet", {}), None)
        return 0

    log("\n  Nothing to do: pass --check (verify masks) or --run (call nnU-Net).")
    log("  Imaging data are not distributed with this repository; see data/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
