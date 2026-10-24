#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 0 — assemble the published input tables into ``data/``.

This is the only script that reads the authors' archive.  It copies the
**deployment version** of every input, writes a SHA-256 manifest, and never
touches any path whose name contains ``副本`` / ``_deprecated`` / ``_deleted``
/ ``回退镜像`` / ``xxx\\`` (see ``docs/OLD_CODE_INVENTORY.md``).

Usage
-----
    python src/00_assemble_data.py [--config config/config.yaml] [--skip-features]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrpca import (  # noqa: E402
    banner,
    ensure_dir,
    load_config,
    log,
    manifest,
    paths,
    sha256,
    write_json,
)

COHORTS = ["train", "test1", "test2", "test_prospective"]
SIGNATURES = [
    "Clinic",
    "Habitat",
    "HabitatH1",
    "HabitatH2",
    "HabitatH3",
    "HabitatH4",
    "HabitatH5",
    "Combined",
]


def _bare(v: str) -> str:
    """Drop an imaging file extension so identifiers share one key space.

    ``label.csv`` stores ``<admission-number>.nii.gz`` while ``Clinic.csv``
    stores the bare number; both denote the same patient.
    """
    s = str(v).strip()
    return s[:-7] if s.lower().endswith(".nii.gz") else s


def _copy(src: Path, dst: Path, records: list, note: str) -> bool:
    if not src.exists():
        log(f"  !! MISSING  {src}")
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    records.append(
        dict(dst=str(dst.name), note=note, src=str(src), bytes=dst.stat().st_size, sha256=sha256(dst))
    )
    return True


# ---------------------------------------------------------------------------
# real_id repair
# ---------------------------------------------------------------------------
# The archived ``scripts/_cases_with_ids.csv`` carries a ``real_id`` column that
# is **row-shifted by one** in the training cohort (and by one in a few
# prospective rows): for 306/340 uniquely identifiable training rows the ID on
# row *i* belongs to row *i+1*.  Trusting it silently mislabels every
# score-row -> patient link in the training cohort and makes the deployed
# training scores look like a broken model (training AUC 0.570 instead of 0.798).
#
# The repair does not guess.  ``scores_highprec.csv`` carries the same ten
# probability columns *with* the study identifier, so each deployed score row
# can be resolved exactly: the seven raw signature probabilities, rounded to the
# two decimals at which the deployed files were released, form a key that is
# unique for all 1,016 patients.  The rebuilt ``real_id`` is then cross-checked
# against the H2/H4 fingerprint and the row is rejected if either disagrees.
_SIG_COLS = ["Clinic", "Habitat", "HabitatH1", "HabitatH2", "HabitatH3", "HabitatH4",
             "HabitatH5"]
# ``cases_with_ids.csv`` labels its rows in Chinese, ``scores_highprec.csv`` in
# English; the repair key must use one vocabulary.
_COHORT_EN = {"训练": "train", "测试1": "test1", "测试2": "test2",
              "前瞻": "test_prospective"}


def _repair_real_id(data_dir: Path) -> tuple[int, int]:
    """Rebuild ``real_id`` in ``cases_with_ids.csv`` from the keyed score export.

    Returns ``(n_repaired, n_disagree_before)``.  Raises if any row cannot be
    resolved uniquely, or if the rebuild does not agree with the H2/H4
    fingerprint — a silent fallback here would reintroduce the defect.
    """
    cwi_path = data_dir / "cases_with_ids.csv"
    hp_path = data_dir / "scores_highprec.csv"
    if not cwi_path.exists() or not hp_path.exists():
        log("  [repair] cases_with_ids.csv or scores_highprec.csv absent — skipped")
        return 0, 0

    cwi = pd.read_csv(cwi_path, encoding="utf-8-sig")
    hp = pd.read_csv(hp_path, encoding="utf-8-sig")
    for df, col in ((cwi, "real_id"), (hp, "ID")):
        df["_k"] = df[col].astype(str).map(_bare).str.strip()
    hp["_grp"] = hp["group"].astype(str).str.strip()

    missing = [c for c in _SIG_COLS if c not in hp.columns]
    if missing:
        raise RuntimeError(f"scores_highprec.csv lacks {missing}")

    # one key table per cohort: 7 probabilities rounded to the released precision
    key: dict[tuple, str] = {}
    for _, r in hp.iterrows():
        k = (r["_grp"],) + tuple(round(float(r[c]), 2) for c in _SIG_COLS)
        if k in key:
            raise RuntimeError(
                f"key collision in scores_highprec.csv for {k}: "
                f"{key[k]} vs {r['_k']}")
        key[k] = r["_k"]

    # the deployed score files, in the same row order as cases_with_ids
    new_id, n_old_ok = [], 0
    for _, r in cwi.iterrows():
        coh = _COHORT_EN.get(str(r["cohort"]).strip(), str(r["cohort"]).strip())
        k = (coh,) + tuple(
            round(float(r[c]), 2) for c in _SIG_COLS if c in cwi.columns)
        if len(k) != len(_SIG_COLS) + 1:
            raise RuntimeError("cases_with_ids.csv lacks the signature columns needed to repair real_id")
        hit = key.get(k)
        if hit is None:
            raise RuntimeError(f"cannot resolve row patient_index={r.get('patient_index')} "
                               f"cohort={r.get('cohort')} — refusing to guess")
        new_id.append(hit)

    old = cwi["_k"].tolist()
    n_old_ok = sum(1 for a, b in zip(old, new_id) if a == b)
    cwi = cwi.drop(columns=["_k"])
    cwi["real_id"] = new_id

    # cross-check 1: every rebuilt ID must exist in model_inputs.csv
    mi = pd.read_csv(data_dir / "model_inputs.csv", encoding="utf-8-sig")
    mi["_k"] = mi["ID"].astype(str).map(_bare).str.strip()
    known = set(mi["_k"])
    unknown = sorted({i for i in new_id if i not in known})
    if unknown:
        raise RuntimeError(f"{len(unknown)} rebuilt real_id values absent from "
                           f"model_inputs.csv: {unknown[:5]}")

    # cross-check 2: the rebuilt ID must be the patient whose H2/H4 the row's
    # own values round to (the fingerprint that exposed the defect)
    mi = mi.set_index("_k")
    fp = {}
    for _, r in mi.iterrows():
        fp.setdefault((round(float(r["H2"]), 2), round(float(r["H4"]), 2)),
                      set()).add(r.name)
    uniq = {k: next(iter(v)) for k, v in fp.items() if len(v) == 1}
    n_fp = n_fp_ok = 0
    for _, r in cwi.iterrows():
        kk = (round(float(r["HabitatH2"]), 2), round(float(r["HabitatH4"]), 2))
        if kk in uniq:
            n_fp += 1
            if uniq[kk] == r["real_id"]:
                n_fp_ok += 1
    if n_fp and n_fp_ok != n_fp:
        raise RuntimeError(f"fingerprint cross-check failed: {n_fp_ok}/{n_fp}")

    cwi.to_csv(cwi_path, index=False, encoding="utf-8-sig")
    log(f"  [repair] real_id rebuilt: {len(new_id)}/{len(new_id)} resolved uniquely; "
        f"{n_old_ok}/{len(new_id)} were already correct "
        f"(fingerprint {n_fp_ok}/{n_fp})")
    return len(new_id), len(new_id) - n_old_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-features", action="store_true", help="skip the ~115 MB habitat feature tables")
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = paths(cfg)
    A = cfg["archives"]
    deploy = Path(A["deploy_root"])
    legacy = Path(A["legacy_code_repo"])
    records: list = []

    banner("00 · assemble data/")
    for key in ("data", "data_scores", "data_features"):
        ensure_dir(P[key])

    # ------------------------------------------------------------------
    # 1. clinical + labels  (Table 1 authoritative sources)
    # ------------------------------------------------------------------
    log("\n[1] clinical variables and cohort labels")
    _copy(Path(A["clinic_csv"]), P["data"] / "clinical.csv", records,
          "Table 1 / Table S15 clinical variables (no imputation)")
    if A.get("clinic_csv_workspace"):
        _copy(Path(A["clinic_csv_workspace"]), P["data"] / "clinical_workspace.csv", records,
              "later 1097-row build — cross-check ONLY, does not reproduce Table S15")
    _copy(Path(A["label_csv"]), P["data"] / "labels.csv", records,
          "ID / label / group for the raw screening set")

    # ---- normalise the label key in place -------------------------------
    # ``label.csv`` stores ``<admission-number>.nii.gz`` while ``Clinic.csv``
    # stores the bare number.  They denote the same patient, so strip the
    # extension once here; every downstream join then uses ONE key space.
    from hrpca import pseudonym as PS

    _lab = pd.read_csv(P["data"] / "labels.csv", dtype=str,
                       keep_default_na=False, encoding="utf-8-sig")
    _raw = _lab["ID"].astype(str)
    _lab["ID"] = [_bare(v) for v in _raw]
    _n_fixed = int((_lab["ID"] != _raw).sum())
    _lab.to_csv(P["data"] / "labels.csv", index=False, encoding="utf-8-sig")
    log(f"    labels.csv  ID normalised (stripped .nii.gz): {_n_fixed} rows")

    # ---- establish the pseudonymisation map ONCE -------------------------
    # The out-of-repo mapping table is authoritative when present; otherwise it
    # is rebuilt from the raw sources (labels.csv is still raw at this point on
    # a first run).  The map is APPLIED later, after every copy: steps [2]–[6]
    # re-import raw identifiers from the archive, so an early replacement would
    # be silently undone.
    from hrpca import pseudonym as PS

    try:
        _MAP = PS.load_map([(A["clinic_csv"], "ID"), (P["data"] / "labels.csv", "ID")])
    except Exception as exc:                                    # pragma: no cover
        log(f"  !! cannot obtain the pseudonymisation map: {exc}")
        return 2
    log(f"    identifier map: {len(_MAP)} numbers -> {min(_MAP.values())}…{max(_MAP.values())}"
        f"  ({'stored' if PS.MAP_JSON.exists() else 'rebuilt'})")

    clinical = pd.read_csv(P["data"] / "clinical.csv", encoding="utf-8-sig")
    # normalise the column names once, so that no downstream script depends on
    # the original spacing / hyphenation.  Both builds are handled:
    #   <archive>/Clinic.csv              volume_tumor, volume_prostate
    #   lunwen3/Clinic.csv           volume_tumor, "prostate volume"
    clinical = clinical.rename(columns={
        "PI-RADS": "PI_RADS",
        "volume_tumor": "target_roi_volume",
        "volume_prostate": "prostate_volume",
        "prostate volume": "prostate_volume",
    })
    clinical["ID"] = clinical["ID"].astype(str).str.replace(".nii.gz", "", regex=False).str.strip()
    clinical.to_csv(P["data"] / "clinical.csv", index=False, encoding="utf-8")
    labels = pd.read_csv(P["data"] / "labels.csv", encoding="utf-8-sig")
    labels["ID"] = labels["ID"].astype(str).str.replace(".nii.gz", "", regex=False).str.strip()
    labels.to_csv(P["data"] / "labels.csv", index=False, encoding="utf-8")
    log(f"    clinical.csv  n = {len(clinical)}  columns = {list(clinical.columns)}")
    log(f"    labels.csv    n = {len(labels)}  groups = {labels['group'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 2. frozen analysis set (n = 1016) and high-precision model inputs
    # ------------------------------------------------------------------
    log("\n[2] frozen analysis set and high-precision model inputs")
    heat_all = pd.read_csv(deploy / "SHAP_output" / "heatmap" / "heatmap_data_all.csv", encoding="utf-8-sig")
    heat_all = heat_all.rename(columns={"habitat": "Habitat", "H1": "HabitatH1", "H2": "HabitatH2",
                                        "H3": "HabitatH3", "H4": "HabitatH4", "H5": "HabitatH5"})
    analysis_set = heat_all[["ID", "group", "label"]].copy()
    analysis_set.to_csv(P["data"] / "analysis_set.csv", index=False, encoding="utf-8")
    log(f"    analysis_set.csv  n = {len(analysis_set)}  pN1 = {int(analysis_set.label.sum())}")
    log(f"      per cohort: {analysis_set.group.value_counts().to_dict()}")

    inputs = heat_all[["ID", "group", "label", "ISUP", "PI_RADS", "HabitatH2", "HabitatH4"]].rename(
        columns={"HabitatH2": "H2", "HabitatH4": "H4"}
    )
    # unrounded probabilities for every signature except the combined model
    hi = heat_all[["ID", "group", "label", "Clinic", "Habitat", "HabitatH1", "HabitatH2",
                   "HabitatH3", "HabitatH4", "HabitatH5", "Combined"]].rename(
        columns={"Combined": "Combined_superseded"}
    )
    hi.to_csv(P["data"] / "scores_highprec.csv", index=False, encoding="utf-8")
    inputs.to_csv(P["data"] / "model_inputs.csv", index=False, encoding="utf-8")
    records.append(dict(dst="scores_highprec.csv", note="16-digit per-signature scores (train AUC source)",
                        src=str(deploy / "SHAP_output" / "heatmap" / "heatmap_data_all.csv"),
                        bytes=(P["data"] / "scores_highprec.csv").stat().st_size,
                        sha256=sha256(P["data"] / "scores_highprec.csv")))
    records.append(dict(dst="model_inputs.csv", note="ID/group/ISUP/PI-RADS/H2/H4 (16-digit)",
                        src=str(deploy / "SHAP_output" / "heatmap" / "heatmap_data_all.csv"),
                        bytes=(P["data"] / "model_inputs.csv").stat().st_size,
                        sha256=sha256(P["data"] / "model_inputs.csv")))
    records.append(dict(dst="analysis_set.csv", note="frozen 1016-patient analysis set",
                        src=str(deploy / "SHAP_output" / "heatmap" / "heatmap_data_all.csv"),
                        bytes=(P["data"] / "analysis_set.csv").stat().st_size,
                        sha256=sha256(P["data"] / "analysis_set.csv")))

    # ------------------------------------------------------------------
    # 3. frozen per-signature scores for the five cohorts
    # ------------------------------------------------------------------
    log("\n[3] frozen per-signature scores")
    for c in COHORTS:
        src = legacy / "derived_data" / f"{c}_scores_label.csv"
        dst = P["data_scores"] / f"{c}.csv"
        if _copy(src, dst, records, f"deployed scores, {c}"):
            d = pd.read_csv(dst, encoding="utf-8-sig")
            log(f"    {c:<18} n = {len(d):>4}  pN1 = {int(d.label.sum()):>3}  AUC(Combined) = "
                f"{_auc(d.label, d.Combined):.6f}")

    # ------------------------------------------------------------------
    # 4. K = 4 / 5 / 7 runs (Table 3 / S3b)
    # ------------------------------------------------------------------
    log("\n[4] K-specific exports")
    ensure_dir(P["data"] / "k_runs")
    kdirs = {4: Path(A["sol21_k4"]) / "figdata", 5: legacy / "derived_data", 7: Path(A["sol21_k7"]) / "figdata"}
    for k, d in kdirs.items():
        src = d / "train_scores_label.csv"
        if not src.exists():
            log(f"    !! K={k}: cannot infer IDs ({src} has no ID column)")
        frames = []
        for c in COHORTS:
            f = d / f"{c}_scores_label.csv"
            if f.exists():
                x = pd.read_csv(f, encoding="utf-8-sig")
                x["cohort"] = c
                frames.append(x)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            col = "patient_index" if "patient_index" in out.columns else "ID"
            out = out.rename(columns={col: "row_index"})
            # K = 4 and K = 7 key on the NIfTI file name (i.e. the same admission
            # number); K = 5 only has a within-cohort row index (1…N).  Strip the
            # extension ONLY where a file name is present, so the K = 5 index is
            # not mangled and step [7] does not treat it as an identifier.
            out["row_index"] = [
                _bare(v) if str(v).endswith(".nii.gz") else str(v)
                for v in out["row_index"].astype(str)
            ]
            dst = P["data"] / "k_runs" / f"K{k}.csv"
            out.to_csv(dst, index=False, encoding="utf-8")
            records.append(dict(dst=f"k_runs/K{k}.csv", note=f"K = {k} frozen scores",
                                src=str(d), bytes=dst.stat().st_size, sha256=sha256(dst)))
            log(f"    K = {k}: {len(out)} rows -> k_runs/K{k}.csv")

    # ------------------------------------------------------------------
    # 5. habitat-level radiomics feature tables
    # ------------------------------------------------------------------
    if not args.skip_features:
        log("\n[5] habitat radiomics feature tables (K = 5)")
        ensure_dir(P["data_features"])
        fdir = Path(A["sol21_k5"]) / "features"
        for h in range(1, 6):
            src = fdir / f"rad_features_h{h}.csv"
            dst = P["data_features"] / f"rad_features_h{h}.csv"
            if _copy(src, dst, records, f"habitat {h} radiomics features"):
                log(f"    h{h}: {dst.stat().st_size / 1e6:.1f} MB")
        for k, d in {4: Path(A["sol21_k4"]), 7: Path(A["sol21_k7"])}.items():
            for h in range(1, k + 1):
                src = d / "features" / f"rad_features_h{h}.csv"
                dst = P["data_features"] / f"k{k}_rad_features_h{h}.csv"
                _copy(src, dst, records, f"K = {k} habitat {h} radiomics features")
    else:
        log("\n[5] habitat feature tables SKIPPED (--skip-features)")

    # ------------------------------------------------------------------
    # 6. K-selection outputs, scanner census, SHAP outputs
    # ------------------------------------------------------------------
    log("\n[6] K selection, the authoritative identifier key, SHAP outputs")
    _copy(Path(A["k_cv10x10_summary"]), P["data"] / "K_CV10x10_summary.csv", records,
          "100-fold repeated CV summary (Table S3)")
    # ★ this file is the only table that links the score files to patient
    #   identifiers — see src/hrpca/linkage.py and docs/OLD_CODE_INVENTORY.md §N4
    _copy(Path(A["cases_with_ids"]), P["data"] / "cases_with_ids.csv", records,
          "★ authoritative key: patient_index -> real_id + cohort")
    _copy(Path(A["fullprec_scores"]), P["data"] / "fullprec_scores.csv", records,
          "deployed scores with identifiers and 16-digit precision")
    # ★ the archived key file is row-shifted in the training cohort — rebuild it
    #   from the keyed score export and refuse to continue if that is impossible
    #   (needs scores_highprec.csv, written in step [1]).
    _n_fixed, _n_changed = _repair_real_id(P["data"])
    if _n_fixed:
        # the file changed on disk, so its manifest entry must be recomputed
        for _r in records:
            if _r["dst"] == "cases_with_ids.csv":
                _p = P["data"] / "cases_with_ids.csv"
                _r["bytes"] = _p.stat().st_size
                _r["sha256"] = sha256(_p)
                _r["note"] = (f"★ authoritative key: patient_index -> real_id + cohort "
                              f"(real_id rebuilt from scores_highprec.csv; "
                              f"{_n_changed} row(s) corrected)")
    shap = deploy / "SHAP_output"
    for rel, note in [
        ("h2h4_final/meta_all.csv", "closed-form SHAP inputs, final combined model"),
        ("h2h4_final/base.csv", "baseline E[f(X)] = -1.1039913"),
        ("h2h4_fusion/fusion_probs_all.csv", "per-model probabilities (20 digits)"),
        ("SHAP_k5_feature_stats.csv", "Table S14: RF-ranked features (27 rows), with mean|SHAP|"),
        ("h2_h4_shap/SHAP_H2_ranking.csv", "H2 feature ranking"),
        ("h2_h4_shap/SHAP_H4_ranking.csv", "H4 feature ranking"),
        ("total_shap/SHAP_habitat_share_train.csv", "habitat share of total mean|SHAP|"),
    ]:
        _copy(shap / rel, P["data"] / "shap" / Path(rel).name, records, note)
    _copy(legacy / "derived_data" / "dl_comparator_results.csv", P["data"] / "dl_comparator_results.csv",
          records, "deep-learning comparator, pooled results (Table S19)")
    _copy(legacy / "derived_data" / "dl_bootstrap_ci.csv", P["data"] / "dl_bootstrap_ci.csv",
          records, "deep-learning comparator bootstrap CIs")

    # ------------------------------------------------------------------
    # 7. pseudonymise EVERYTHING that was just copied
    #
    # ★ This must run **after** all copies: steps [2]–[6] re-import raw
    #   identifiers from the archive, so a replacement placed earlier would be
    #   silently undone.  Admission numbers are direct identifiers and must not
    #   reach a public repository.
    # ------------------------------------------------------------------
    log("\n[7] pseudonymisation (admission numbers -> S0001…)")
    _tot = 0
    _files = 0
    for _f in sorted(P["data"].rglob("*.csv")):
        _df = pd.read_csv(_f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        _n = sum(h for _, h in PS.apply_to_frame(_df, _MAP, PS.ID_COLUMNS))
        if _n:
            _df.to_csv(_f, index=False, encoding="utf-8-sig")
            _tot += _n
            _files += 1
    log(f"    {_tot} identifier values in {_files} files -> {min(_MAP.values())}…{max(_MAP.values())}")

    # ------------------------------------------------------------------
    # 8. manifest
    # ------------------------------------------------------------------
    # written AFTER pseudonymisation, and re-hashing every destination, so the
    # manifest never fingerprints an un-pseudonymised table
    _dst = P["data"]
    for r in records:
        p = _dst / r["dst"]
        if p.is_file():
            r["bytes"] = p.stat().st_size
            r["sha256"] = sha256(p)
    write_json(
        dict(
            generated_by="src/00_assemble_data.py",
            n_files=len(records),
            total_bytes=sum(r["bytes"] for r in records),
            pseudonymised=True,
            identifier_map_location="OUTSIDE the repository (never commit)",
            entries=records,
        ),
        P["data"] / "MANIFEST.json",
    )
    log(f"\n    {len(records)} files, {sum(r['bytes'] for r in records) / 1e6:.1f} MB")
    log(f"    manifest -> {P['data'] / 'MANIFEST.json'}")

    # ------------------------------------------------------------------
    # 8. sanity check
    # ------------------------------------------------------------------
    banner("data/ sanity check")
    ok = True
    for c in COHORTS:
        d = pd.read_csv(P["data_scores"] / f"{c}.csv", encoding="utf-8-sig")
        exp = cfg["cohorts"][c]
        good = len(d) == exp["n"] and int(d.label.sum()) == exp["pn1"]
        ok &= good
        log(f"  {c:<18} n = {len(d):>4} (expect {exp['n']:>4})   pN1 = {int(d.label.sum()):>3} "
            f"(expect {exp['pn1']:>3})   {'OK' if good else 'MISMATCH'}")
    log(f"\n  overall: {'OK' if ok else 'PROBLEM'}")
    return 0 if ok else 1


def _auc(y, p) -> float:
    import numpy as np

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (pos.size * neg.size))


if __name__ == "__main__":
    raise SystemExit(main())
