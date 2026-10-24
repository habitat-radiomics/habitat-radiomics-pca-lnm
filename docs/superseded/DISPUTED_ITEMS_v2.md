> # ⛔ RETRACTED — DO NOT USE
>
> **This document is superseded.**  Its three Table S15 "改稿建议" (n 377 → 374;
> PI-RADS OR 1.611 → 1.565; PSA density OR 1.118 → 1.123) were all **wrong**:
> they came from the 1,097-row `lunwen3/Clinic.csv`.  With the authoritative
> `<archive>/Clinic.csv`, **Table S15 reproduces 14/14 cells with n = 377 and
> PI-RADS OR 1.611** — nothing needs changing.
>
> ► The current, correct conclusions are in
> [`../ARCHIVE_PROVENANCE.md`](../ARCHIVE_PROVENANCE.md).
> This file is kept only as a record of the method error that produced it.

---

# 争议项复核（第 2 版）— 方法已修正

**日期**：2026-10-16 20:0x
**提交人**：代码库重写会话
**铁律**：`<submission-package>` **全程只读**，未修改任何文件

---

## 0 · 先修正方法：我上一版为什么会错

### 根因：读了 workspace 里的**过期对照稿**

| 文件 | 角色 | `Table 4` 是否存在 | 最后修改 |
|---|---|---|---|
| `<submission-package>\05_源文件\Manuscript_FINAL.md` | ✅ **权威正文** | ❌ **不存在**（只有 Table 1/2/3） | 2026-10-16 19:38 |
| `<submission-package>\02_补充材料\Supplementary_Material.md` | ✅ **权威补充** | ❌（只有 S1–S21） | 2026-10-16 19:56 |
| `<deploy-root>\投稿版_Manuscript_FINAL_EN.md` | ❌ **工作区旧稿（我误用）** | ✅ 存在（第 238–246 行） | **2026-10-09** |

我引用的 `.336` 出自 **`lunwen3\投稿版_Manuscript_FINAL_EN.md` 第 244 行**
（该旧稿的"Table 4" Test 2 行）。已确认：**权威稿中不存在 Table 4**，
且权威 Table S5 与 Table S10 的 Test 2 DeLong **均已为 .371**。

**新方法（已固化为代码）**：
* `src/hrpca/paper.py` —— **运行时**从权威正文/补充材料解析全部论文值（正则解析 markdown 表）；
* `verify/check_disputed_items.py` —— 用解析值对照复算值；
* **不再使用任何硬编码的"论文值"**（上一版 `src/hrpca/gold.py` 里的 `DELONG_TABLE4` 已改名为
  `DELONG_TABLE4_STALE` 并标注"过期稿、勿用"；11 项 DeLong 差异条目已从 `KNOWN_DISCREPANCIES` 删除）；
* **源码守门（source guard）**：`verify/check_paper_numbers.py` 每次运行都会断言
  *"被投稿正文里不存在 Table 4"*，若发现即 FAIL —— 从机制上防止再次误读过期稿。

`paper.py` 的解析结果（自检）：正文表 `['Table 1','Table 2','Table 3']`；
补充材料 `Table S1–S21`（其中**无 Table S14**，已合并进 S10 脚注）。

---

## ① DeLong —— 结论：**复算与权威稿完全一致，无差异项**

**复算命令**：`python verify/check_disputed_items.py`

| 来源 | 值 |
|---|---|
| **权威 Table S5**（运行时解析） | 见下表"论文"列 |
| **权威 Table S10**（运行时解析） | Test 2 DeLong = **.371** |
| **部署分数**（`data/scores/*.csv`） | 见下表"复算"列 |
| **归档流水线导出**（`derived_data/*_Delong.csv`） | 独立第二来源 |

| 队列 | vs Clinical 论文 / 复算 / 流水线 | vs H4 | vs H2 | vs All-habitat |
|---|---|---|---|---|
| Training | <.001 / 0.00007 / 0.000065 | .012 / .0119 / .012369 | <.001 / .00001 | .002 / .0019 |
| Test 1 | **.255 / .2553 / .255307** | <.001 / .0003 | <.001 / .0001 | <.001 / .0000 |
| **Test 2** | **.371 / .3711 / .371131** | .003 / .0023 | .002 / .0013 | <.001 / .0001 |
| Prospective | .003 / .0031 / .003105 | .202 / .2016 | .066 / .0664 | .003 / .0029 |
| Pooled external | **.022 / .0218** | <.001 / .0000 | <.001 / .0000 | <.001 / .0000 |

**结论**：**24/24 格一致**（部署分数与流水线导出在 16/16 格一致，差 < 1e-4）。
**无任何 DeLong 差异项。我上一版的 11 项 DeLong"差异"全部作废。**

---

## ② Table S9 —— **论文值来自两个不同来源（这是真实情况，但需注明口径）**

| 项 | 内容 |
|---|---|
| **论文位置** | 补充材料 `## **Table S9. Calibration details by cohort**` |
| **论文表注（原文）** | *"Values for the **clinical and combined models** were computed on the **saved scores of the deployed models** (the same models reported in Table 2); values for the **habitat signatures** are from the **analysis pipeline**."* |
| **依据文件（复算）** | a) `data/scores/<cohort>.csv`（部署分数）<br>b) `<deploy-root>/code_repository/derived_data\hosmer_all.csv`（归档流水线导出） |
| **ID 连接方式** | 同表内逐患者计算，**不需要任何连接**（H-L 只用本队列的 label + 本列分数） |
| **复算脚本** | `python verify/check_disputed_items.py`（② 段），或用 `src/07_calibration.py` |

### 逐格判定（复算值 = 部署分数；对照列 = 归档 `hosmer_all.csv`）

| 签名 | 队列 | 论文 | 部署分数 | 归档 hosmer_all | 判定 |
|---|---|---|---|---|---|
| **Habitat（27 特征）** | Training | 0.052 | 0.6502 | **0.0517** | ✅ = 归档 |
| | Test 1 | 0.089 | 0.0000 | **0.0886** | ✅ = 归档 |
| | Test 2 | 0.208 | 0.0000 | **0.2080** | ✅ = 归档 |
| | Prospective | 0.830 | 0.0013 | **0.8303** | ✅ = 归档 |
| **Habitat H1** | Training | 0.004 | 0.0741 | **0.0042** | ✅ = 归档 |
| | Test 1 | 0.003 | 0.0000 | **0.0026** | ✅ = 归档 |
| | Test 2 | 0.974 | 0.0000 | **0.9741** | ✅ = 归档 |
| | Prospective | 0.231 | 0.2625 | **0.2308** | ✅ = 归档 |
| **Habitat H2** | Training | 0.065 | 0.6467 | **0.0647** | ✅ = 归档 |
| | Test 1 | 0.314 | 0.0000 | **0.3142** | ✅ = 归档 |
| | Test 2 | 0.214 | 0.0000 | **0.2145** | ✅ = 归档 |
| | Prospective | 0.001 | 0.0050 | **0.0011** | ✅ = 归档 |
| **Habitat H3** | Training | 0.003 | 0.3336 | **0.0035** | ✅ = 归档 |
| | Test 1 | 0.360 | 0.0000 | **0.3597** | ✅ = 归档 |
| | Test 2 | 0.828 | 0.0004 | **0.8283** | ✅ = 归档 |
| | Prospective | 0.083 | 0.2785 | **0.0827** | ✅ = 归档 |
| **Habitat H4** | Training | 0.007 | 0.0611 | **0.0071** | ✅ = 归档 |
| | Test 1 | 0.102 | 0.0000 | **0.1017** | ✅ = 归档 |
| | Test 2 | 0.115 | 0.0000 | **0.1150** | ✅ = 归档 |
| | Prospective | 0.262 | 0.6628 | **0.2623** | ✅ = 归档 |
| **Habitat H5** | Training | 0.012 | 0.1583 | **0.0116** | ✅ = 归档 |
| | Test 1 | 0.011 | 0.0000 | **0.0113** | ✅ = 归档 |
| | Test 2 | 0.687 | 0.0000 | **0.6870** | ✅ = 归档 |
| | Prospective | 0.124 | 0.1361 | **0.1241** | ✅ = 归档 |
| **Combined** | Training | 0.351 | **0.3514** | 0.0669 | ✅ = 部署分数 |
| | Test 1 | 0.0002 | **0.0002** | 0.0000 | ✅ = 部署分数 |
| | Test 2 | 0.018 | **0.0177** | 0.0298 | ✅ = 部署分数 |
| | Prospective | 0.073 | **0.0729** | 0.0103 | ✅ = 部署分数 |
| **Clinical** | Training | **0.012** | 0.0343 | 0.0658 | ❌ **两个来源都不是** |
| | Test 1 | 0.0004 | 0.0007 | 0.0003 | ⚠️ 两个都接近（差 ≤0.0004） |
| | Test 2 | **0.091** | 0.0350 | 0.0474 | ❌ **两个来源都不是** |
| | Prospective | 0.380 | 0.1162 | 0.3752 | ✅ = 归档（差 0.005） |

### 结论与建议

1. **生境各行（Habitat、H1–H5）共 20 格：全部逐格等于归档 `hosmer_all.csv`（≤0.0008）**，
   与论文表注"habitat signatures are from the analysis pipeline"**完全自洽** →
   **建议不改**（表注已说明口径）。
2. **Combined 行 4 格：全部等于部署分数** → 与表注一致 → **建议不改**。
3. **Clinical 行：Test 1 与 Prospective 可归入归档来源，但 Training（论文 .012 vs 归档 .0658）
   与 Test 2（论文 .091 vs 归档 .0474 / 部署 .0350）两个来源都解释不了** ⚠️
   → **建议单独核对该行的原始来源**；这是本项唯一的真实疑点。
4. **我上一版说"H1 的 test1/test2 互换"—— 错误，特此作废**
   （0.974 本就属 H1 的 test2，归档也是 0.9741）。

---

## ③ Table S15 的 n —— **结论：真实差异，且不止 n**

| 项 | 内容 |
|---|---|
| **论文位置** | 补充材料 `## **Table S15. Univariable and multivariable logistic regression for pN1 (training cohort, n = 377 with complete data)**` |
| **论文表头** | n = **377** |
| **论文脚注（原文）** | *"Seven training patients lacked ISUP grade group and a further 14 lacked PSA, so **377 of 398** patients had complete data for the multivariable model."* |
| **依据文件** | `Clinic.csv`（临床变量）+ `label.csv`（group/label） |
| **ID 连接方式** | **`label.csv` 的 `ID`（去掉 `.nii.gz`）←→ `Clinic.csv` 的 `ID`**（纯数字）。**这两个文件本身共享数字 ID，不需要 `_cases_with_ids.csv`**；`group == 'train'` 取自 `label.csv`。 |
| **复算脚本** | `python verify/check_disputed_items.py`（③ 段）；表格另存 `results/tables/disputed_items_verification.csv` |

### n 的复算

| 步骤 | 例数 |
|---|---|
| `label.csv` group=train | 407 |
| ∩ `Clinic.csv` | **395** |
| 7 个变量全非缺失 | **374** |
| 缺失明细 | ISUP **7**、PSA **15**、PSAD **15**、Age/volume 0 |

论文脚注写"缺 ISUP 7、缺 PSA 14、完整 377"。**我复算：缺 ISUP 7（吻合）、缺 PSA 15（差 1）、完整 374（差 3）。**
→ **口径一致（同为 complete-case），差异来自 `Clinic.csv` 的不同版本**（当前版本的 PSA/PSAD 各多缺 1 例，且训练队列为 395 而非 398 例）。

### 逐变量对照（n = 374）

| 变量 | 论文单因素 OR (95% CI) | **复算单因素** | 论文多因素 OR | **复算多因素** |
|---|---|---|---|---|
| Age | 1.003 (0.972–1.035) | **1.0018 (0.971–1.034)** ✅ | 1.001 (0.968–1.036) | **0.9997 (0.966–1.035)** ✅ |
| **PI-RADS** | 1.681 (1.134–2.493) | **1.6778 (1.130–2.492)** ✅ | **1.611 (1.048–2.477)** | **1.5653 (1.015–2.413)** ❌ |
| PSA | 1.006 (1.002–1.009) | **1.0058 (1.002–1.009)** ✅ | 1.004 (0.999–1.009) | **1.0041 (0.999–1.009)** ✅ |
| ISUP | 2.047 (1.609–2.603) | **2.0465 (1.608–2.604)** ✅ | 1.891 (1.476–2.422) | **1.8896 (1.474–2.423)** ✅ |
| Target-ROI volume | 0.995 (0.978–1.013) | **0.9947 (0.977–1.012)** ✅ | 1.013 (0.976–1.051) | **1.0104 (0.973–1.049)** ✅ |
| Prostate volume | 0.996 (0.986–1.006) | **0.9955 (0.986–1.005)** ✅ | 0.989 (0.967–1.011) | **0.9903 (0.968–1.013)** ✅ |
| **PSA density** | 1.118 (1.031–1.212) | **1.1231 (1.033–1.221)** ❌ | 1.019 (0.930–1.117) | **1.0209 (0.929–1.122)** ✅ |

**14 格中 12 格吻合。**

### 我也测试过是否能换口径凑出 377 / 1.611 —— 不能

已穷举（脚本可复跑）：7 变量、去掉 target-ROI volume、去掉 prostate volume、去掉两者、
去掉 PSA、去掉 PSAD、去掉 PSA+PSAD、仅 ISUP+PI-RADS、ISUP+PI-RADS+PSA，共 9 种变量集 × 2 个队列范围。

| 组合 | n | ISUP 多因素 | PI-RADS 多因素 |
|---|---|---|---|
| **全 7 变量** | **374** | **1.8896** ✅ | **1.5653** |
| 去掉 PSA | 374 | 1.9144 | 1.5844 |
| 去掉 PSA density | 374 | 1.8915 | 1.5577 |
| 仅 ISUP+PI-RADS | 388 | 1.9507 | 1.3219 |
| 全队列（非训练） | 1047 | 2.2095 | 1.8817 |

**没有任何组合同时得到 n = 377 与 PI-RADS OR = 1.611。**

### 建议

| 位置 | 论文现值 | 复算值 | 建议 |
|---|---|---|---|
| Table S15 表头 n | 377 | **374** | ⭕ **改 374**（或注明"377 in the earlier clinical release"） |
| Table S15 PI-RADS 多因素 OR | 1.611 (1.048–2.477) | **1.565 (1.015–2.413)** | ⭕ **改用复算值**；这是唯一实质变动 |
| Table S15 PSA density 单因素 OR | 1.118 (1.031–1.212) | **1.123 (1.033–1.221)** | ⭕ 建议改（末位） |
| Table S15 其余 11 格 | — | — | ❌ **不需改** |
| 正文 3.9 节"PI-RADS OR 1.611" | 1.611 | **1.565** | ⭕ 同步 |

---

## ④ Table S16a 临床行 —— **结论：无需改（用与 Table 2 相同的来源即可完全复现）**

| 项 | 内容 |
|---|---|
| **论文位置** | 补充材料 `## **Table S16. … alternative clinical comparators**`，第 2 行 `**ISUP grade group + PI-RADS (primary comparator)**` |
| **论文值** | 0.706 / 0.795 / 0.770 / 0.689（train / test1 / test2 / prospective） |
| **依据文件** | a) **部署分数**：`data/scores/<cohort>.csv` 的 `Clinic` 列；train 用 16 位源 `data/scores_highprec.csv`<br>b) 重新拟合：`heatmap_data_all` 的 ISUP/PI-RADS（= `data/model_inputs.csv`）+ `Clinic.csv` |
| **ID 连接方式** | (b) 用 **`model_inputs.csv` 的数字 ID ←→ `Clinic.csv` 的 ID**；逐患者相关另用 `data/cases_with_ids.csv` 的 `patient_index → real_id`（仅 test 队列可靠） |
| **复算脚本** | `python verify/check_disputed_items.py`（④ 段） |

### 结果

| 队列 | 论文 | **部署分数 AUC** | 判定 | 重新拟合 AUC | 重新拟合 vs 部署：逐患者 |
|---|---|---|---|---|---|
| Training | 0.706 | **0.7064**（16 位源） | ✅ | 0.7018 | —（train 的 Clinic 只有 2 位小数，相关无意义） |
| Test 1 | 0.795 | **0.7952** | ✅ | 0.8036 | **概率相关 1.0000，最大差 0.0031** |
| Test 2 | 0.770 | **0.7697** | ✅ | 0.7713 | **概率相关 1.0000，最大差 0.0031** |
| Prospective | 0.689 | **0.6890** | ✅ | 0.6801 | 相关 0.9008，最大差 0.2836 |

拟合式（训练 n=398）：`logit(p) = −4.969726 + 0.656099·ISUP + 0.329798·PI-RADS`

### 结论与建议

* **该行 4 格全部可复现，不需改稿。**
* 部署的 `Clinic` 概率在 test1/test2 上与 ISUP+PI-RADS 的 logistic **秩完全相同**
  （相关 1.0000，最大差 0.0031）→ **它就是 ISUP+PI-RADS 的 logistic 输出**。
* Prospective 的差（0.2836）来自该队列 `Clinic` 被舍入到 2 位小数，不是模型差异。
* **我上一版说"部署分数不是普通 logistic"—— 错误，特此作废**（当时也是因为误用了过期稿/错误连接）。

---

## ⑤ 附录：Table 1（供参考，因第 3 条曾要求）

| 项 | 内容 |
|---|---|
| **论文位置** | 正文 `**Table 1.**` |
| **依据文件** | `data/model_inputs.csv`（= `SHAP_output/heatmap/heatmap_data_all.csv`，含 ID/group/label/ISUP/PI-RADS）+ `Clinic.csv`（连续变量） |
| **ID 连接方式** | **两表都有纯数字 ID，直接 `on="ID"` 连接**；**严禁按行位次配对**（positional 相关 ≈0.02） |
| **复算脚本** | `python verify/check_disputed_items.py`（⑥ 段） |

| 项目 | 结果 |
|---|---|
| 队列 n（398/339/189/90） | ✅ 4/4 |
| **PI-RADS 3/4/5**（3 行 × 4 队列） | ✅ **12/12 完全一致** |
| **ISUP 1/2/4/5**（4 行 × 4 队列） | ✅ **16/16 完全一致** |
| **ISUP 3 训练队列** | ❌ 论文 **76** vs 复算 **83**（论文 ISUP 合计 **391** ≠ 398；缺 ISUP 的 7 例被折进 grade 3） |
| 连续变量中位数 | 15/20 格在 0.2 内；5 格超出（PSA train 33.5 vs 32.9；Age pros 71.5 vs 72.0；prostate volume train 37.5 vs 37.7、test2 33.4 vs 33.6） |

→ **投稿会话的"Table 1 可复现"判断基本正确**（28 个分类格中 27 格完全一致）；
唯一疑点是 **ISUP grade 3 训练队列 76 vs 83**。我上一版的"32/76/40/27/11"是
**在错误连接（`analysis_set` ∩ `Clinic.csv`，11 例缺临床）下算出的**，**已作废**。

---

## ⑥ 复核入口（可独立复算）

```powershell
cd <deploy-root>\habitat-radiomics-pca-lnm

# ① 从权威稿解析全部论文值（自检）
python -X utf8 src/hrpca/paper.py

# ② 六项逐一复核（论文值 vs 复算值 vs 依据文件 vs 连接方式）
python -X utf8 verify/check_disputed_items.py
#   → results/tables/disputed_items_verification.csv（含 item/cell/published/computed/
#      source/id_join/verdict/note 八列）

# ③ 全库一键核验
python -X utf8 verify/check_paper_numbers.py -v
```

`disputed_items_verification.csv` 的列设计就是为了让你**逐格独立复算**：
`published`（运行时解析）、`computed`、`source`（具体文件）、`id_join`（具体连接方式）。

---

## ⑦ 本版结论一览

| # | 项 | 上一版结论 | **本版结论** |
|---|---|---|---|
| ① | DeLong | 11 格差异 | **24/24 一致，0 差异**（作废） |
| ② | Table S9 生境行 | "H1 test1/test2 互换" | **无互换**；20 格全等于归档 `hosmer_all.csv`，符合表注口径，**建议不改**；**仅 Clinical 行 2 格来源不明** |
| ③ | Table S15 的 n | 374 + PI-RADS 1.565 | **维持**：n=**374**、PI-RADS 多因素 **1.565**；另 PSA density 单因素末位需改；11 格不动 |
| ④ | Table S16a | "部署分数不是 logistic" | **撤销**：部署分数就是 ISUP+PI-RADS 的 logistic，**4 格全对，不建议改** |
| ⑤ | `base.csv` | "是 1009 行均值" | **撤销**：是 **398 例训练队列**均值（差 3×10⁻⁶），**文件正确** |
| ⑥ | Table 1 | 45 格差异 | **撤销**：27/28 分类格一致；仅 **ISUP grade 3 训练队列 76 vs 83** |

**最终需要改稿的只剩 3 处**：
1. **Table S15 表头 n：377 → 374**
2. **Table S15 + 正文 3.9：PI-RADS 多因素 OR 1.611 → 1.565 (1.015–2.413)**
3. **Table S15：PSA density 单因素 OR 1.118 → 1.123 (1.033–1.221)**
   （另：Table 1 的 ISUP grade 3 训练队列 76 vs 83 建议核对；Table S9 的 Clinical 行 2 格建议核对来源）
