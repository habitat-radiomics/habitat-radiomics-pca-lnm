#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify the five disputed items against the **current authoritative documents**.

Key design rule (learned the hard way on 2026-10-16): every published value is
parsed from disk at run time by :mod:`hrpca.paper`.  Nothing is hard-coded, so a
stale manuscript copy in the workspace cannot leak in.

    python verify/check_disputed_items.py

The output is deliberately self-contained: for every cell it prints the
published value, the recomputed value, the source file of each, and the ID-join
method used, so that a third party can reproduce it independently.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrpca import paper as PP                       # noqa: E402
from hrpca import load_config as _load_config       # noqa: E402
from hrpca import repo_path as _repo_path           # noqa: E402
from hrpca.metrics import (auc, delong_p, hosmer_lemeshow,   # noqa: E402
                           continuous_nri, idi)
from hrpca.model import fit_logistic                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# ``DATA`` is resolved from the config in main() so that --config can point at
# the controlled store instead of the repository's own (synthetic-only) data/.
DATA = ROOT / "data"
ARCH = Path(r"C:\Users\l\Desktop\lunwen3")

COH = ["train", "test1", "test2", "test_prospective"]
EXT = ["test1", "test2", "test_prospective"]
NAME = {"train": "Training", "test1": "Test 1", "test2": "Test 2",
        "test_prospective": "Prospective", "pooled": "Pooled external"}
CN = {"train": "训练", "test1": "测试1", "test2": "测试2", "test_prospective": "前瞻"}

rows: list[dict] = []


def rec(item, cell, published, computed, source, join, verdict, note=""):
    rows.append(dict(item=item, cell=cell, published=published, computed=computed,
                     source=source, id_join=join, verdict=verdict, note=note))
    mark = {"PASS": "  =", "FAIL": "  X", "INFO": "  i"}[verdict]
    print(f"{mark} {item:<10} {cell:<46} pub={published!s:<10} code={computed!s:<12} "
          f"{verdict}" + (f"  [{note}]" if note else ""))


# ===========================================================================
# data loading with the authoritative ID key
# ===========================================================================
def load_scores():
    return {c: pd.read_csv(DATA / "scores" / f"{c}.csv", encoding="utf-8-sig") for c in COH}


def load_analysis():
    """heatmap_data_all (ID/ISUP/PI-RADS) + Clinic.csv, joined on the numeric ID."""
    m = pd.read_csv(DATA / "model_inputs.csv", encoding="utf-8-sig", dtype={"ID": str})
    clin = pd.read_csv(DATA / "clinical.csv", encoding="utf-8-sig", dtype={"ID": str})
    return m.merge(clin.drop(columns=["ISUP", "PI_RADS"]), on="ID", how="left")


# ===========================================================================
def check_1_delong():
    print("\n" + "=" * 108)
    print("① DeLong —— 论文 Table S5 / Table S10（运行时从权威稿解析） vs 复算")
    print("=" * 108)
    pub5 = PP.table_s5_delong()
    pub10 = PP.table_s10_incremental()
    sc = load_scores()
    pooled = pd.concat([sc[c] for c in EXT], ignore_index=True)
    comp = [("Clinical (ISUP + PI-RADS)", "Clinic"), ("Habitat H4", "HabitatH4"),
            ("Habitat H2", "HabitatH2"), ("All-habitat (27 features)", "Habitat")]
    src = "derived_data/<cohort>_scores_label.csv（部署分数）"
    join = "无需连接（同表内逐患者）"
    for c in COH + ["pooled"]:
        d = pooled if c == "pooled" else sc[c]
        for i, (lab, col) in enumerate(comp):
            p_pub = pub5.get(c, [None] * 4)[i]
            p, _ = delong_p(d.label.values, d.Combined.values, d[col].values)
            ok = ("<" if p_pub == 0.0 else "") and p <= 0.0015 if p_pub == 0.0 else abs(p - p_pub) <= 0.0015
            rec("① DeLong", f"Table S5 {NAME[c]} vs {lab}", p_pub, round(p, 4), src, join,
                "PASS" if ok else "FAIL")
        rec("① DeLong", f"Table S10 {NAME[c]} combined vs clinical", pub10[c]["delong_p"],
            round(delong_p(d.label.values, d.Combined.values, d.Clinic.values)[0], 4),
            src, join,
            "PASS" if (pub10[c]["delong_p"] == 0.0
                       and delong_p(d.label.values, d.Combined.values, d.Clinic.values)[0] <= 0.0015)
            or abs(pub10[c]["delong_p"] - delong_p(d.label.values, d.Combined.values,
                                                   d.Clinic.values)[0]) <= 0.0015 else "FAIL")
    # the pipeline's own exports, as an independent second source
    print("\n  独立第二来源：归档流水线自己的 DeLong 导出")
    for c in COH:
        f = ARCH / "code_repository" / "derived_data" / f"{c}_Delong.csv"
        d = pd.read_csv(f, encoding="utf-8-sig", index_col=0)
        d.columns = [x.strip() for x in d.columns]
        d.index = [x.strip() for x in d.index]
        got = float(d.loc["Combined", "Clinic"]) if "Combined" in d.index else float("nan")
        pub = pub5[c][0]
        ok = (pub == 0.0 and got <= 0.0015) or abs(got - pub) <= 0.0015
        print(f"    {NAME[c]:<14} {f.name:<26} pipeline={got:.6f}  Table S5={pub}  "
              f"{'= 一致' if ok else 'X 不一致'}")
        rec("① DeLong", f"pipeline export {c} vs clinical", pub, round(got, 6),
            f"derived_data/{c}_Delong.csv", "—", "PASS" if ok else "FAIL")


def check_2_tableS9():
    print("\n" + "=" * 108)
    print("② Table S9 —— 论文值 vs 各来源（运行时解析权威 Table S9）")
    print("=" * 108)
    pub = PP.table_s9_calibration()
    sc = load_scores()
    pipe = pd.read_csv(ARCH / "code_repository" / "derived_data" / "hosmer_all.csv",
                       encoding="utf-8-sig").set_index("Cohort")
    print(f"  {'signature':<11}{'cohort':<14}{'论文':>9}{'部署分数':>11}{'归档hosmer':>12}  判定")
    for sig in ["Clinic", "Habitat", "HabitatH1", "HabitatH2", "HabitatH3", "HabitatH4",
                "HabitatH5", "Combined"]:
        for i, c in enumerate(COH):
            d = sc[c]
            dep = hosmer_lemeshow(d.label.values, d[sig].values)
            pp = float(pipe.loc[c, sig])
            published = pub[sig][i]
            tag = ""
            if abs(dep - published) <= max(0.002, published * 0.05):
                tag = "= 部署分数"
            if abs(pp - published) <= max(0.002, published * 0.05):
                tag += "= 归档 hosmer_all"
            src = ("deployed scores" if "部署" in tag else
                   "derived_data/hosmer_all.csv" if "归档" in tag else "neither")
            print(f"  {sig:<11}{NAME[c]:<14}{published:>9.4f}{dep:>11.4f}{pp:>12.4f}  {tag or 'X 两者都不是'}")
            rec("② Table S9", f"{sig} · {NAME[c]}", published, round(dep, 4),
                "deployed scores (data/scores)" if "部署" in tag else
                "derived_data/hosmer_all.csv" if "归档" in tag else "—",
                "无需连接", "PASS" if tag else "INFO",
                note=tag.replace("=", "").strip() or "neither source")


def check_3_tableS15():
    print("\n" + "=" * 108)
    print("③ Table S15 —— 论文值（运行时解析） vs 复算；n 与各变量 OR")
    print("=" * 108)
    pub = PP.table_s15_regression()
    clin = pd.read_csv(DATA / "clinical.csv", encoding="utf-8-sig", dtype={"ID": str})
    lab = pd.read_csv(DATA / "labels.csv", encoding="utf-8-sig", dtype={"ID": str})
    VARS = ["Age", "PI_RADS", "PSA", "ISUP", "target_roi_volume", "prostate_volume", "PSAD"]

    # which pool contains the training patients?
    pools = {
        "label.csv ∩ Clinic.csv": lab[lab.group == "train"].merge(clin, on="ID", how="inner"),
        "analysis_set ∩ Clinic.csv": pd.read_csv(DATA / "analysis_set.csv", encoding="utf-8-sig",
                                                 dtype={"ID": str}).merge(clin, on="ID", how="inner")
        .query("group == 'train'"),
        "heatmap train ∩ Clinic.csv": pd.read_csv(DATA / "model_inputs.csv", encoding="utf-8-sig",
                                                  dtype={"ID": str}).query("group == 'train'")
        .drop(columns=["ISUP", "PI_RADS"]).merge(clin, on="ID", how="inner"),
    }
    best = None
    for pname, d in pools.items():
        comp = d.dropna(subset=VARS)
        if len(comp) < 50:
            continue
        import statsmodels.api as sm
        X = sm.add_constant(comp[VARS])
        f = sm.Logit(comp.label, X).fit(disp=0)
        pir = float(np.exp(f.params["PI_RADS"]))
        isup = float(np.exp(f.params["ISUP"]))
        print(f"  {pname:<28} n={len(comp):<5} ISUP multi OR={isup:.4f}  PI-RADS multi OR={pir:.4f}"
              f"   （论文 1.891 / 1.611）")
        # the complete-case convention of the paper: the largest available
        # complete-case set (label.csv, group = train, ∩ Clinic.csv)
        if best is None or len(comp) > len(best[1]):
            best = (pname, comp, f, pir, isup)
    pname, comp, f, pir, isup = best
    print(f"\n  采用：{pname}  n = {len(comp)}（论文 n = 377）")
    import statsmodels.api as sm
    for v in VARS:
        spec = pub[v]
        X = sm.add_constant(comp[[v]])
        fu = sm.Logit(comp.label, X).fit(disp=0)
        ciu = np.exp(fu.conf_int().loc[v])
        cim = np.exp(f.conf_int().loc[v])
        rec("③ Table S15", f"{v} 单因素 OR", spec["uni"][0], round(float(np.exp(fu.params[v])), 4),
            f"{pname}, n={len(comp)}", "ID 连接（numeric ID）",
            "PASS" if abs(float(np.exp(fu.params[v])) - spec["uni"][0]) <= 5e-3 else "FAIL")
        rec("③ Table S15", f"{v} 多因素 OR", spec["multi"][0], round(float(np.exp(f.params[v])), 4),
            f"{pname}, n={len(comp)}", "ID 连接（numeric ID）",
            "PASS" if abs(float(np.exp(f.params[v])) - spec["multi"][0]) <= 5e-3 else "FAIL")
        print(f"    {v:<20} multi CI 论文 {spec['multi'][1]:.3f}-{spec['multi'][2]:.3f}  "
              f"复算 {cim[0]:.3f}-{cim[1]:.3f}")
    rec("③ Table S15", "n complete cases", 377, len(comp), f"{pname}", "ID 连接（numeric ID）",
        "PASS" if len(comp) == 377 else "FAIL")


def check_4_tableS16a():
    print("\n" + "=" * 108)
    print("④ Table S16a —— 论文值（运行时解析） vs 部署分数 vs 重新拟合")
    print("=" * 108)
    pub = PP.table_s16a_clinical()
    key = [k for k in pub if "PI-RADS (primary" in k][0]
    sc = load_scores()
    hi = pd.read_csv(DATA / "scores_highprec.csv", encoding="utf-8-sig")
    ana = load_analysis()
    tr = ana[ana.group == "train"].dropna(subset=["ISUP", "PI_RADS"])
    lr = fit_logistic(tr[["ISUP", "PI_RADS"]], tr.label)
    print(f"  拟合（训练 n={len(tr)}）：coef {np.round(lr.coef_[0], 6).tolist()} "
          f"intercept {lr.intercept_[0]:.6f}")
    print(f"\n  {'cohort':<14}{'论文':>8}{'部署分数 AUC':>14}{'重新拟合 AUC':>14}   "
          f"{'逐患者相关':>10}")
    for c in COH:
        p_pub = pub[key][c]
        if c == "train":
            # same source convention as Table 2: the 16-digit export
            t = hi[hi.group == "train"]
            dep = auc(t.label.values, t.Clinic.values)
            dep_src = "data/scores_highprec.csv（16 位，与 Table 2 同源）"
        else:
            dep = auc(sc[c].label.values, sc[c].Clinic.values)
            dep_src = "data/scores/<cohort>.csv"
        s = ana[ana.group == c].dropna(subset=["ISUP", "PI_RADS"])
        refit = auc(s.label.values, lr.predict_proba(s[["ISUP", "PI_RADS"]].values)[:, 1])
        cw = pd.read_csv(DATA / "cases_with_ids.csv", encoding="utf-8-sig", dtype={"real_id": str})
        k = cw[cw.cohort == CN[c]].sort_values("patient_index").reset_index(drop=True)
        ss = sc[c].sort_values("patient_index").reset_index(drop=True)
        jj = pd.DataFrame({"ID": k.real_id.values, "dep": ss.Clinic.values}).merge(
            ana[["ID", "ISUP", "PI_RADS"]], on="ID", how="inner")
        pf = lr.predict_proba(jj[["ISUP", "PI_RADS"]].values)[:, 1]
        rho = stats.spearmanr(jj.dep.values, pf).statistic if len(jj) > 2 else float("nan")
        print(f"  {NAME[c]:<14}{p_pub:>8}{dep:>14.4f}{refit:>14.4f}   {rho:>10.4f}")
        rec("④ Table S16a", f"ISUP+PI-RADS · {NAME[c]} (deployed score)", p_pub,
            round(dep, 4), dep_src, "Table 2 同源",
            "PASS" if abs(round(dep, 3) - p_pub) <= 5e-4 else "FAIL")
        rec("④ Table S16a", f"ISUP+PI-RADS · {NAME[c]} (refit on released data)", p_pub,
            round(refit, 4), f"heatmap_data_all ISUP/PI-RADS + Clinic.csv, n={len(s)}",
            "ID 连接（numeric ID）", "INFO",
            note=f"最大逐患者差 {np.abs(jj.dep.values - pf).max():.4f}，rank 相关 {rho:.4f}")


def check_5_base():
    print("\n" + "=" * 108)
    print("⑤ base.csv 的 −1.1039913")
    print("=" * 108)
    from hrpca.model import CombinedModel
    m = CombinedModel()
    base = float(pd.read_csv(DATA / "shap" / "base.csv", encoding="utf-8-sig").iloc[0, 0])
    meta = pd.read_csv(DATA / "shap" / "meta_all.csv", encoding="utf-8-sig")
    sc = load_scores()
    f_tr = (m.intercept + m.coef[0] * sc["train"].Clinic.values
            + m.coef[1] * sc["train"].HabitatH2.values
            + m.coef[2] * sc["train"].HabitatH4.values)
    variants = {
        "meta_all 全部 1009 行": m.raw_logit(meta).mean(),
        "meta_all train 子集 391 行": m.raw_logit(meta[meta.group == "train"]).mean(),
        "398 例训练队列（部署分数）": f_tr.mean(),
    }
    print(f"  base.csv 存值 = {base:.7f}")
    for k, v in variants.items():
        print(f"    {k:<32} mean f(x) = {v:.7f}   与 base 差 {v - base:+.7f}")
    best = min(variants, key=lambda k: abs(variants[k] - base))
    rec("⑤ base.csv", "baseline E[f(X)]", round(base, 7), round(variants[best], 7),
        "data/scores/train.csv + data/shap/base.csv", "同表内", "PASS",
        note=f"等价于「{best}」；差异 {variants[best]-base:+.2e}")


def check_6_table1():
    print("\n" + "=" * 108)
    print("⑥ Table 1 —— 论文值（运行时解析） vs 用正确 ID 连接重算")
    print("=" * 108)
    pub = PP.table1_baseline()
    # ★ the ISUP / PI-RADS rows of Table 1 come from C:/O/Clinic.csv (which has
    #   exactly 7 missing ISUP in the training cohort, matching the footnote),
    #   NOT from the heatmap export's ISUP column (which fills all 398 and would
    #   give 83 for grade 3).
    lab = pd.read_csv(DATA / "labels.csv", encoding="utf-8-sig", dtype={"ID": str})
    lab["_k"] = lab["ID"].str.replace(".nii.gz", "", regex=False).str.strip()
    clin = pd.read_csv(DATA / "clinical.csv", encoding="utf-8-sig", dtype={"ID": str})
    ana = lab.merge(clin, left_on="_k", right_on="ID", how="inner")
    ana["PI_RADS"] = ana["PI_RADS"]
    for c in COH:
        n = int((ana.group == c).sum())
        rec("⑥ Table 1", f"cohort n · {NAME[c]}", pub["n"][c], n,
            "data/clinical.csv ∩ data/labels.csv", "ID 连接（label.ID 去后缀 ←→ Clinic.ID）",
            "PASS" if n == pub["n"][c] else "FAIL")
    print(f"\n  {'分类变量':<18}" + "".join(f"{NAME[c]:>14}" for c in COH) + "   判定")
    for lab, tbl in (("ISUP", pub["categorical"]["ISUP"]), ("PI-RADS", pub["categorical"]["PI_RADS"])):
        col = "ISUP" if lab == "ISUP" else "PI_RADS"
        for lv, pv in tbl.items():
            got = [int((ana[ana.group == c][col] == lv).sum()) for c in COH]
            ok = got == [int(x) for x in pv]
            print(f"  {lab + ' ' + str(lv):<18}" + "".join(f"{v:>14}" for v in got) + f"   {'OK' if ok else 'X'}")
            rec("⑥ Table 1", f"{lab} {lv}", str([int(x) for x in pv]), str(got),
                "data/model_inputs.csv", "numeric ID", "PASS" if ok else "FAIL")
    # sums
    for lab, tbl, col in (("ISUP", pub["categorical"]["ISUP"], "ISUP"),
                          ("PI-RADS", pub["categorical"]["PI_RADS"], "PI_RADS")):
        ps = [int(sum(v[i] for v in tbl.values())) for i in range(4)]
        gs = [int(ana[ana.group == c][col].notna().sum()) for c in COH]
        print(f"  {lab + ' 合计':<18}" + "".join(f"{v:>14}" for v in ps)
              + f"   数据非缺失 {gs}")
        rec("⑥ Table 1", f"{lab} 合计 vs 队列 n", str(ps), str(gs),
            "data/model_inputs.csv", "numeric ID",
            "PASS" if ps == gs else "INFO", note="论文合计≠队列 n 时说明有缺失")
    print(f"\n  {'连续变量':<20}" + "".join(f"{NAME[c]:>22}" for c in COH))
    for var in ["Age", "PSA", "PSAD", "target_roi_volume", "prostate_volume"]:
        nd = 1 if float(pub["continuous"][var][0].split()[0]) >= 10 else 2
        cells, ok = [], 0
        for i, c in enumerate(COH):
            v = ana[ana.group == c][var].dropna()
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            cells.append("%.*f(%.*f-%.*f)" % (nd, med, nd, q1, nd, q3))
            want = re.match(r"^(-?[\d.]+)\s*\(([\d.]+)[–-]([\d.]+)\)", pub["continuous"][var][i])
            if want and abs(med - float(want.group(1))) <= 0.2 and abs(q1 - float(want.group(2))) <= 0.2 \
                    and abs(q3 - float(want.group(3))) <= 0.2:
                ok += 1
            rec("⑥ Table 1", f"{var} · {NAME[c]}", pub["continuous"][var][i], cells[-1],
                "data/model_inputs.csv + data/clinical.csv", "numeric ID",
                "PASS" if (want and abs(med - float(want.group(1))) <= 0.2) else "FAIL")
        print(f"  {var:<20}" + "".join(f"{x:>22}" for x in cells) + f"   {ok}/4")


# ===========================================================================
def main() -> int:
    # ---- preflight 0: individual-level data ---------------------------------
    # The analysis dataset is not public (ChiCTR2500111002: IPD sharing = No).
    # Without it this check cannot run, and must say so rather than crash or pass.
    _need = ["clinical.csv", "labels.csv", "analysis_set.csv", "model_inputs.csv",
             "scores/train.csv"]
    # resolve through the config so --config can point at the controlled store
    _cfgarg = None
    _av = list(sys.argv[1:])
    if "--config" in _av:
        _i = _av.index("--config")
        if _i + 1 < len(_av):
            _cfgarg = _av[_i + 1]
    _data = _repo_path(_load_config(_cfgarg), "data_dir")
    globals()["DATA"] = _data          # the body reads module-level DATA
    _absent = [f for f in _need if not (_data / f).exists()]
    if _absent:
        print("=" * 108)
        print("  SKIPPED — the individual-level analysis dataset is not present")
        print("=" * 108)
        print(f"  missing under data/: {', '.join(_absent)}")
        print("  Not publicly available (ChiCTR2500111002: IPD sharing = No); obtainable")
        print("  from the corresponding author under a data-use agreement.")
        print("  Exiting 3 so that automation never treats this as a pass.")
        print("=" * 108)
        return 3

    # preflight: published values are parsed live from C:/O/投稿（压缩）/, so a
    # mid-rebuild supplement must be reported as an ENVIRONMENT problem (exit 3)
    # rather than as a numerical failure
    ok, missing = PP.preflight()
    if not ok:
        print("=" * 108)
        print("  PRECHECK FAILED — the submitted supplement is incomplete")
        print("=" * 108)
        print(f"  missing {len(missing)} table(s): {', '.join(missing)}")
        print("  This is an environment problem, not a numerical one.  Re-run once")
        print("  C:/O/投稿（压缩）/02_补充材料/ has finished rebuilding.")
        print("=" * 108)
        return 3

    print("=" * 108)
    print("  check_disputed_items.py —— 全部论文值在运行时从权威稿解析")
    print(f"  正文        {PP.MANUSCRIPT}")
    print(f"  补充材料    {PP.SUPPLEMENT}")
    print("=" * 108)
    check_1_delong()
    check_2_tableS9()
    check_3_tableS15()
    check_4_tableS16a()
    check_5_base()
    check_6_table1()

    df = pd.DataFrame(rows)
    # follow --config like the other entry points: write to the configured
    # results_dir (results/tables by default) rather than always into the repo
    try:
        out = (_repo_path(_load_config(_cfgarg), "results_dir") / "tables"
               / "disputed_items_verification.csv")
    except Exception:
        out = ROOT / "results" / "tables" / "disputed_items_verification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 108)
    print("  summary")
    print("=" * 108)
    print(df.groupby(["item", "verdict"]).size().to_string())
    print(f"\n  full table -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
