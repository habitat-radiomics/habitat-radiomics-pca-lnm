#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03 — habitat-level radiomics feature extraction (PyRadiomics + region remapping).

Per-habitat features are extracted with PyRadiomics from the habitat masks, and
the five habitat tables are aggregated into a single candidate matrix by the
**region-remapping** rule: a feature extracted in habitat *h* is kept under the
name ``<feature>_h{h}``, so that the same nominal feature in different habitats
becomes a different candidate.  This is why the candidate set contains 1,197
columns rather than the ~107 features of a single PyRadiomics configuration.

Two levels of data are relevant:

* ``data/features/rad_features_h{1..5}.csv`` — the habitat-level tables that
  ship with this repository (n = 1016 per habitat, 1,197 features each).  These
  reproduce **Table S2** (habitat profiles, ``--profiles``) and **Table S14**
  (Table S14: features ranked by random-forest importance, 27 rows, with the archived statistics).
* the *candidate matrix used for feature selection* — built here by
  :func:`build_candidate_matrix`.  Note that the values of that matrix are not
  identical to the shipped tables (the pipeline extracted features on the
  production masks), so re-running LASSO from the shipped tables does not
  reproduce the retained set; see ``docs/VERIFICATION_REPORT.md`` finding F8.

Imaging data are not distributed, so extraction itself is only runnable by a
reader with data access.

Usage
-----
    python src/03_feature_extraction.py --profiles          # Table S2
    python src/03_feature_extraction.py --candidates        # build the 1,197-column matrix
    python src/03_feature_extraction.py --extract --masks DIR --images DIR --out DIR
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrpca import banner, ensure_dir, load_config, log, paths, write_json  # noqa: E402
from hrpca import gold as G                                                  # noqa: E402

N_HABITATS = 5
CANDIDATE_FEATURES = 1197

# The three base image types, eight wavelet sub-bands and four LoG scales used by
# the production run.  These lists reproduce the delivered tables exactly:
#   Original            105 = 91 non-shape + 14 shape
#   Wavelet   8 x 91 =  728
#   LoG       4 x 91 =  364
#   ---------------------------
#   total               1,197
IMAGE_TYPES = ["Original", "Wavelet", "LoG"]
WAVELET_BANDS = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
LOG_SIGMAS = [2.0, 3.0, 4.0, 5.0]

#: PyRadiomics 3.0.1 feature names per class, transcribed from the delivered
#: 1,197-column tables so that the expected name universe can be enumerated
#: without PyRadiomics installed.  ``shape`` is computed on the mask and
#: therefore exists only for the ``Original`` image type.
FEATURE_NAMES = {
    "firstorder": (
        "10Percentile", "90Percentile", "Energy", "Entropy",
        "InterquartileRange", "Kurtosis", "Maximum",
        "MeanAbsoluteDeviation", "Mean", "Median", "Minimum", "Range",
        "RobustMeanAbsoluteDeviation", "RootMeanSquared", "Skewness",
        "TotalEnergy", "Uniformity", "Variance",
    ),
    "glcm": (
        "Autocorrelation", "ClusterProminence", "ClusterShade",
        "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage",
        "DifferenceEntropy", "DifferenceVariance", "Id", "Idm", "Idmn", "Idn",
        "Imc1", "Imc2", "InverseVariance", "JointAverage", "JointEnergy",
        "JointEntropy", "MaximumProbability", "SumEntropy", "SumSquares",
    ),
    "glrlm": (
        "GrayLevelNonUniformityNormalized", "GrayLevelNonUniformity",
        "GrayLevelVariance", "HighGrayLevelRunEmphasis", "LongRunEmphasis",
        "LongRunHighGrayLevelEmphasis", "LongRunLowGrayLevelEmphasis",
        "LowGrayLevelRunEmphasis", "RunEntropy",
        "RunLengthNonUniformityNormalized", "RunLengthNonUniformity",
        "RunPercentage", "RunVariance", "ShortRunEmphasis",
        "ShortRunHighGrayLevelEmphasis", "ShortRunLowGrayLevelEmphasis",
    ),
    "glszm": (
        "GrayLevelNonUniformityNormalized", "GrayLevelNonUniformity",
        "GrayLevelVariance", "HighGrayLevelZoneEmphasis",
        "LargeAreaEmphasis", "LargeAreaHighGrayLevelEmphasis",
        "LargeAreaLowGrayLevelEmphasis", "LowGrayLevelZoneEmphasis",
        "SizeZoneNonUniformityNormalized", "SizeZoneNonUniformity",
        "SmallAreaEmphasis", "SmallAreaHighGrayLevelEmphasis",
        "SmallAreaLowGrayLevelEmphasis", "ZoneEntropy", "ZonePercentage",
        "ZoneVariance",
    ),
    "gldm": (
        "DependenceEntropy", "DependenceNonUniformityNormalized",
        "DependenceNonUniformity", "DependenceVariance",
        "GrayLevelNonUniformity", "GrayLevelVariance",
        "HighGrayLevelEmphasis", "LargeDependenceEmphasis",
        "LargeDependenceHighGrayLevelEmphasis",
        "LargeDependenceLowGrayLevelEmphasis", "LowGrayLevelEmphasis",
        "SmallDependenceEmphasis", "SmallDependenceHighGrayLevelEmphasis",
        "SmallDependenceLowGrayLevelEmphasis",
    ),
    "ngtdm": ("Busyness", "Coarseness", "Complexity", "Contrast", "Strength"),
    "shape": (
        "Elongation", "Flatness", "LeastAxisLength", "MajorAxisLength",
        "Maximum2DDiameterColumn", "Maximum2DDiameterRow",
        "Maximum2DDiameterSlice", "Maximum3DDiameter", "MeshVolume",
        "MinorAxisLength", "Sphericity", "SurfaceArea", "SurfaceVolumeRatio",
        "VoxelVolume",
    ),
}

#: feature classes that exist for every image type (``shape`` does not)
NON_SHAPE_CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")

# The PyRadiomics configuration, following IBSI guidance.
PYRADIOMICS_SETTINGS = {
    "binWidth": 5,
    "resampledPixelSpacing": [2, 2, 2],
    "interpolator": "sitkBSpline",
    "normalize": True,
    "normalizeScale": 100,
    "voxelArrayShift": 300,
    "geometryTolerance": 1e-4,
    "force2D": False,
    "imageTypes": IMAGE_TYPES,
    "sigma": LOG_SIGMAS,          # LoG scales, in mm
    "featureClass": ["firstorder", "shape", "glcm", "glrlm", "glszm",
                     "gldm", "ngtdm"],
}


def _image_prefixes(image_types=None, bands=None, sigmas=None) -> list:
    """``(prefix, classes)`` pairs implied by the imageType / wavelet / sigma settings."""
    image_types = IMAGE_TYPES if image_types is None else image_types
    bands = WAVELET_BANDS if bands is None else bands
    sigmas = LOG_SIGMAS if sigmas is None else sigmas
    out = []
    for it in image_types:
        if it == "Original":
            out.append(("original", NON_SHAPE_CLASSES + ("shape",)))
        elif it == "Wavelet":
            out.extend((f"wavelet_{b}", NON_SHAPE_CLASSES) for b in bands)
        elif it == "LoG":
            for s in sigmas:
                out.append((f"log_sigma_{str(s).replace('.', '_')}_mm_3D",
                            NON_SHAPE_CLASSES))
        else:
            raise ValueError(f"unsupported image type: {it!r}")
    return out


def enumerate_feature_names(image_types=None, bands=None, sigmas=None) -> list:
    """Every raw feature name the current imageType / wavelet / sigma settings produce."""
    names = []
    for prefix, classes in _image_prefixes(image_types, bands, sigmas):
        for cls in classes:
            names.extend(f"{prefix}_{cls}_{f}" for f in FEATURE_NAMES[cls])
    return names


def check_model_coverage(model_features, image_types=None, bands=None,
                         sigmas=None) -> tuple:
    """Split model feature names (``_h<N>`` stripped) into ``(covered, missing)``."""
    have = set(enumerate_feature_names(image_types, bands, sigmas))
    want = [re.sub(r"_h[1-5]$", "", str(x).strip()) for x in model_features]
    covered = [w for w in want if w in have]
    missing = [w for w in want if w not in have]
    return covered, missing


# ---------------------------------------------------------------------------
def feature_columns(df: pd.DataFrame, habitat: int) -> list:
    suffix = f"_h{habitat}"
    return [c for c in df.columns if c.endswith(suffix)]


def build_candidate_matrix(feature_dir: Path, analysis_set: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the five habitat tables into the 1,197-column candidate matrix."""
    frames = []
    for h in range(1, N_HABITATS + 1):
        d = pd.read_csv(feature_dir / f"rad_features_h{h}.csv", encoding="utf-8-sig",
                        low_memory=False)
        d["_id"] = d["ID"].astype(str).str.replace(".nii.gz", "", regex=False).str.strip()
        keep = ["_id"] + feature_columns(d, h)
        frames.append(d[keep].set_index("_id"))
    matrix = pd.concat(frames, axis=1)
    matrix = matrix.loc[matrix.index.isin(set(analysis_set["ID"].astype(str)))]
    return matrix


def habitat_profiles(feature_dir: Path, analysis_set: pd.DataFrame) -> tuple:
    """Table S2 (a) means and (b) cross-habitat z-scores (ddof = 0)."""
    ids = set(analysis_set["ID"].astype(str))
    means, zscores = {}, {}
    for feat, col in G.TABLE_S2_FEATURE_COLUMNS.items():
        vals = []
        for h in range(1, N_HABITATS + 1):
            d = pd.read_csv(feature_dir / f"rad_features_h{h}.csv", encoding="utf-8-sig",
                            low_memory=False, usecols=["ID", f"{col}_h{h}"])
            d["_id"] = d["ID"].astype(str).str.replace(".nii.gz", "", regex=False).str.strip()
            d = d[d["_id"].isin(ids)]
            vals.append(float(d[f"{col}_h{h}"].mean()))
        means[feat] = vals
        a = np.asarray(vals)
        # ★ the z-scores are computed from the FULL-PRECISION means and keep the
        #   table's own wording (no "(heterogeneity)" stripping): rounding the
        #   means to 4 dp first changes Uniformity by 0.01, and a shortened
        #   label would no longer match panel (b).
        zscores[feat] = list((a - a.mean()) / a.std(ddof=0))
    return means, zscores


def extract_pyradiomics(masks: Path, images: Path, out: Path, settings: dict) -> None:
    """Run PyRadiomics per habitat mask (requires the imaging data)."""
    import radiomics
    from radiomics import featureextractor

    ensure_dir(out)
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    rows = []
    for mask in sorted(masks.glob("*_h*.nii.gz")):
        m = re.search(r"_h(\d)\.nii", mask.name)
        if not m:
            continue
        habitat = int(m.group(1))
        image = images / mask.name.replace(f"_h{habitat}", "")
        if not image.exists():
            log(f"    !! no image for {mask.name}")
            continue
        res = extractor.execute(str(image), str(mask))
        row = {"ID": image.name, "habitat": habitat}
        row.update({k: v for k, v in res.items() if not k.startswith("diagnostics_")})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "rad_features_raw.csv", index=False, encoding="utf-8")
    log(f"    {len(df)} habitat rows -> {out / 'rad_features_raw.csv'}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profiles", action="store_true", help="recompute Table S2")
    ap.add_argument("--candidates", action="store_true", help="build the candidate matrix")
    ap.add_argument("--extract", action="store_true", help="run PyRadiomics (needs images)")
    ap.add_argument("--list-features", action="store_true",
                    help="enumerate the feature names implied by the imageType settings")
    ap.add_argument("--model-features", default=None,
                    help="CSV of the model's feature names; check that this "
                         "configuration produces all of them")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    banner("03 · habitat radiomics feature extraction")
    log(f"  PyRadiomics settings: {PYRADIOMICS_SETTINGS}")

    if args.list_features or args.model_features:
        names = enumerate_feature_names()
        log(f"\n  imageTypes    : {', '.join(IMAGE_TYPES)}")
        log(f"  wavelet bands : {', '.join(WAVELET_BANDS)}")
        log(f"  LoG sigmas    : {', '.join(f'{s:g}' for s in LOG_SIGMAS)}")
        log(f"  enumerated    : {len(names)} feature names "
            f"(expected {CANDIDATE_FEATURES})")
        ok = len(names) == CANDIDATE_FEATURES
        if not ok:
            log(f"  !! this configuration produces {len(names)} names, "
                f"the delivered tables have {CANDIDATE_FEATURES}")
        if args.model_features:
            mf = pd.read_csv(args.model_features, encoding="utf-8-sig")
            col = "feature_name" if "feature_name" in mf.columns else mf.columns[0]
            covered, missing = check_model_coverage(mf[col])
            log(f"  model features: {len(covered)}/{len(covered) + len(missing)} covered")
            for m in missing:
                log(f"    !! not produced by this configuration: {m}")
            ok = ok and not missing
        log(f"  -> {'OK' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.extract:
        if not (args.masks and args.images and args.out):
            log("  --extract needs --masks, --images and --out")
            return 2
        extract_pyradiomics(Path(args.masks), Path(args.images), Path(args.out), PYRADIOMICS_SETTINGS)
        return 0

    analysis_set = pd.read_csv(P["data"] / "analysis_set.csv", encoding="utf-8-sig", dtype={"ID": str})

    if args.profiles or not args.candidates:
        means, zs = habitat_profiles(P["data_features"], analysis_set)
        rows = []
        for feat, vals in means.items():
            rows.append(dict(feature=feat, **{f"H{h+1}": round(v, 4) for h, v in enumerate(vals)}))
        tab_a = pd.DataFrame(rows)
        log("\n  Table S2(a) — mean feature value per habitat (n = 1016)")
        log(tab_a.to_string(index=False))
        # agreement with the published panel
        worst = 0.0
        for _, r in tab_a.iterrows():
            paper = G.TABLE_S2A[r["feature"]]
            worst = max(worst, max(abs(r[f"H{i+1}"] - paper[i]) for i in range(5)))
        log(f"\n  max |recomputed − published| over panel (a): {worst:.4f}")
        tab_a.to_csv(P["results_tables"] / "tableS2a_habitat_profiles.csv", index=False,
                     encoding="utf-8-sig")

        rows_b = []
        for feat, vals in zs.items():
            rows_b.append(dict(feature=feat, **{f"H{h+1}": round(v, 2) for h, v in enumerate(vals)}))
        tab_b = pd.DataFrame(rows_b)
        log("\n  Table S2(b) — z-scores across habitats (ddof = 0)")
        log(tab_b.to_string(index=False))
        tab_b.to_csv(P["results_tables"] / "tableS2b_habitat_zscores.csv", index=False,
                     encoding="utf-8-sig")
        # ★ the z-scores above are computed from the FULL-PRECISION panel (a)
        #   means, not from the 4-decimal values written to the CSV.  Rounding
        #   first changes Uniformity by 0.01 and is the reason the table and
        #   Fig. S7 previously disagreed for three rows.
        if zs:
            first = next(iter(zs))
            log(f"    z from full-precision means (panel (a) values in the CSV are rounded to 4 dp)")
    if args.candidates:
        m = build_candidate_matrix(P["data_features"], analysis_set)
        out = P["data"] / "candidate_features.csv"
        m.to_csv(out, index=False, encoding="utf-8")
        log(f"\n  candidate matrix: {m.shape} -> {out}")
        log(f"  paper reports {CANDIDATE_FEATURES} candidate features after region remapping")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
