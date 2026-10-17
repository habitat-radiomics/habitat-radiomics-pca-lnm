# 步骤 2 · 旧代码清点（部署版 vs 旧版）

**清点日期**：2026-10-16
**目的**：上传 GitHub 前，逐个文件标出「部署版 ✓（可用）」与「旧版 ✗（禁用）」，并给出真实路径。
**依据**：`C:\O\投稿（压缩）\10_投稿清单\权威数据源索引.md` + `开工须知_代码库重写.md` 第三节。

---

## 一、目录层面的版本判定

| 目录 | 判定 | 理由 |
|---|---|---|
| `C:\Users\l\Desktop\lunwen3\` | ✅ **部署版根**（现行分析目录） | 「开工须知」指定的脚本目录（`scripts\`）+ `figures_submission\` + `code_repository\derived_data` 均在此；`SHAP_output\` 高精度数据在此 |
| `C:\Users\l\Desktop\lunwen3\code_repository\` | ✅ **部署版代码库**（将被重写取代） | `derived_data\metrics_all.csv` 与论文 Table 2 完全一致（32/32 AUC 已复核，见下） |
| `C:\O\_完整备份_20261011_001849\01_工作目录\` | ✅ 同上内容的完整备份（只读参照） | 与 lunwen3 同源；`SHAP_output`、`sol21`、`results` 齐备 |
| `C:\O\sol21. 生境分析-(多)单中心-不同区域对比-临床 - 副本` | ❌ **旧版** | 4 变量 nomogram（PI_RADS/ISUP/H2/H4）；Combined 训练 AUC 0.795 ✗（论文 0.798） |
| `C:\O\sol7. 传统组学-多中心-多模态 - 模态对比 -临床` | ⚠️ **旧版（仅 WTR 行参照）** | nomogram 仅 ISUP+ADC；Table S6 最终值需注明口径 |
| `C:\O\xxx\sol21…` | ❌ 旧版 | 第 4 个时期快照，行数不同 |
| `C:\O\K4\`、`C:\O\K7\` | ✅ **K=4 / K=7 专用复现目录** | Table 3 / S3b 的权威源（各自 `figdata`） |
| `C:\O\深度学习建模\sol12_DL*` | ✅ DL 比较器（`_18` 为部署配置） | Table S19 / S2x |
| `C:\O\_完整备份_20261011_001849\04_回退镜像\` | ❌ 回退镜像 | 含旧版；仅作灾难恢复 |
| `C:\O\1Data\` | ✅ 原始影像（ADC/DWI/T2WI/crop/clu） | 不可公开 |

---

## 二、脚本层面的清点（旧库 `code_repository\`，共 22 个 analysis + 5 个 deep_learning + 15 个 notebook）

### 2.1 `analysis\`（Python，可复用逻辑）

| 脚本 | 判定 | 说明 / 真实数据源 |
|---|---|---|
| `recompute_tableS2_habitat_profiles.py` | ⚠️ **逻辑可用、路径是旧版** | 第 6 行指向 `sol21…- 副本`（❌旧版）→ 必须改指 **部署版 features**。且脚本内 `PUB`（论文期望值）为 2026-10-14 之前的旧值（Signal mean H1 3.134 等），与现行 Table S2（3.113/2.848/10.944）**不一致** ✗ → 以**重算值**为准 |
| `recompute_K_new.py` | ✅ 逻辑可用 | K=4/5/7 外部泛化；需指向 `C:\O\K4`、`C:\O\K7`、部署版 `figdata` |
| `recalibration_dca.py` | ✅ 逻辑可用（Table S9/S13/S17 口径） | 用 `*_scores_label.csv`（2 位小数）；H-L = 10 等频分箱，dof = 箱数−2；Cox 重校准 = 队列内 logit 逻辑回归 |
| `sens_52_corrected.py` | ✅ 逻辑可用（Table S18/S20） | 52 例手工修正清单内嵌；注意用**冻结模型**与**重拟合**两种口径 |
| `compare_clinical_comparators.py` | ⚠️ 路径是旧版 | 参考 `sol21…\data\clinical.csv`（含插补 ✗）→ 改为 `Clinic.csv` + `label.csv` |
| `rebuild_27feature_model.py` | ✅ 逻辑可用（Table S14 / 0.864 复核） | 27 特征 LASSO 重建 |
| `verify_tableS14_retained_features.py` | ✅ | 保留特征 5/7/7/5/3 生境分布 |
| `field_strength_subgroup.py` / `_bootstrap.py` | ✅ | Table 4 / Table 5 场强亚组（1.5T 0.849 n=45 vs 3.0T 0.852 n=562, P=0.980） |
| `extract_dicom_acquisition_params.py`、`scanner_census.py`、`map_patients_to_scanners.py`、`scanner_inventory_by_cohort.py` | ✅ | Table S1（29 型号 / 4 厂商；仁济 580、安医前瞻 51） |
| `per_vendor_auc_exploratory.py` | ✅ | 探索性 |
| `final_matched_compare.py`、`new_clinic_aucs2.py` | ✅ | WTR 匹配比较（Table S6b） |
| `diagnose_test1_cal.py` | ✅ | Test1 校准斜率 0.94 / 预测 18.6% vs 观察 13.3% |
| `recal_decision_impact.py` | ✅ | Table S17(b)(c) 决策影响（原始 vs 重校准概率尺度） |
| `recompute_tableS4_tiny_habitat.py`、`redraw_FigS1_tiny_habitat.py` | ✅ | Table S4 / Fig. S1 |
| `redo_nomogram.py`、`redraw_figures.py` | ⚠️ 绘图 | 旧版图（Fig. S4 用旧分数 ✗）→ 必须用部署分数重画 |

### 2.2 `deep_learning\`

| 脚本 | 判定 |
|---|---|
| `01_make_cv10_splits.py` / `02_train_cv10.py` | ✅ 训练与划分（需 GPU + 原始影像） |
| `03_dl_analysis.py` | ✅ **无需重训**即可复现 Table 2 DL 行、S19、S19b（读导出特征） |
| `04_rebuild_crop.py` | ✅ 预处理 |
| `00_data_integrity_notes.md` | ✅ 4 项完整性更正（Table S20） |

### 2.3 `notebooks\`（OnekeyAI 平台，Step0–Step6.3）

| Notebook | 判定 |
|---|---|
| `Step0–Step6.3`（15 个） | ⚠️ **平台执行记录**，`onekey_algo` 核心为 Pyarmor 混淆 ✗ 不可审阅 → 代码库**保留为可追溯记录**，但不作为一键复现路径 |

### 2.4 归档中**不可复用 / 已废弃**的脚本

| 位置 | 原因 |
|---|---|
| `lunwen3\scripts\`（2,914 个 py） | 绝大多数是**改稿/审计/画图**一次性脚本，与复现无关；只挑选其中真正做统计的（`recompute_S3d.py`、`_audit_s3d.py`、`_audit_k_recompute.py` 等）作为口径参照 |
| `lunwen3\_deprecated\`、`_deleted\` | 已废弃 |
| `lunwen3\_audit_*`（2026-10-14） | 审计用临时脚本 |

---

## 三、已复核的关键一致性（本次清点新证据）

| 核对项 | 结果 |
|---|---|
| `derived_data\metrics_all.csv` vs 论文 Table 2 | ✅ **32/32 AUC 全部一致** |
| `heatmap_data_all.csv`（高精度）重算 7 签名 × 4 队列 | ✅ 与 `metrics_all.csv` **32/32 一致**（Clinic train 0.706432→0.706；Habitat train 0.863923→**0.864**）；**唯 Combined 列例外** ✗（旧模型 0.795） |
| 队列规模 | ✅ train 398（118 pN1）/ test1 339（45）/ test2 189（18）/ prospective 90（19）→ 合计 **1016（200 pN1，19.7%）** |
| `*_scores_label.csv` 精度 | ⚠️ 2 位小数确认（重算 Clinic train = 0.704101 ✗，真值 0.706432） |

### ⚠️ 本次清点发现的**新问题**（3 项）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| N1 | **Table S2 的「权威源目录」被标为旧版** ✗ | `开工须知_代码库重写.md` 第 30 行写 `sol21…\features\rad_features_h1–h5.csv`，但磁盘上**存在两个 sol21**：`- 副本`（❌旧版）与部署版（`lunwen3\sol21…\features\`） | 若照抄路径 → Table S2 会算错；**本仓库改用部署版 features** |
| N2 | **`recompute_tableS2_habitat_profiles.py` 内嵌的论文期望值与现行 Table S2 不符** ✗ | 脚本 `PUB` 中 H1 Signal mean = 3.134 / H2 2.921 / GLCM contrast H2 10.940；现行补充材料为 3.113 / 2.848 / 10.944 | 该脚本的「一致性对照」输出会误报；**本仓库以重算值为准**，并单列重算 vs 论文对照 |
| N3 | **`derived_data\*_scores_label.csv` 无 `ID` 列，只有 `patient_index`** ⚠️ | 实际表头：`patient_index,label,Clinic,…,Combined` | 旧文档（`data_dictionary.md`）称有 `ID` 列 → 已过时 |

---

## 三之二、★ N4 根问题：`Clinic.csv` 与 `label.csv` 之间没有共同 ID —— **连接路径已找到**

### 问题陈述

| 文件 | ID 形态 | 例 |
|---|---|---|
| `Clinic.csv` | 纯数字字符串 | `S0001` |
| `label.csv` | NIfTI 文件名 | `S0346.nii.gz` |
| `derived_data\*_scores_label.csv` | 仅组内行号 | `patient_index` = 1…398 |
| `SHAP_output\heatmap\heatmap_data_all.csv` | 纯数字 ID + group + label | `S0411` |

四者**两两没有共同键**。这正是 Table 1 一度算错、以及我一度误判
「Table 1 不可复现」「部署 Clinic 不是 ISUP+PI-RADS 的 logistic」的根因。

### 唯一权威连接键

```
C:\Users\l\Desktop\lunwen3\scripts\_cases_with_ids.csv          （1,016 行）
  patient_index, label, Clinic, Habitat, HabitatH1..H5, Combined,
  real_id, cohort, ck, s_HabitatH2, s_HabitatH4, dom, margin
```

### 连接链（已逐项验证）

```
derived_data/<cohort>_scores_label.csv
      │  (patient_index)      ← 8 个分数列与 _cases_with_ids 逐字节相同
      ▼
_cases_with_ids.csv  ──(real_id)──►  Clinic.csv
                     ──(real_id)──►  label.csv              (.nii.gz → group/label)
                     ──(real_id)──►  heatmap_data_all.csv
                     ──(real_id)──►  _fullprec_scores.csv
                     ──(real_id)──►  heatmap_model_inputs_all.csv
```

### 验证证据

| 检查项 | 结果 |
|---|---|
| `(cohort, patient_index)` 唯一 / `real_id` 唯一 | ✅ 0 重复 / 0 重复 |
| `real_id` ∈ `label.csv` | ✅ **1016 / 1016** |
| `real_id` ∈ `Clinic.csv` | ✅ 1005 / 1016（11 例无临床记录） |
| `real_id` ∈ `heatmap_data_all.csv` | ✅ 1016 / 1016 |
| 8 个分数列 vs `derived_data/*_scores_label.csv` | ✅ 四队列逐字节相同 |
| ID 配对正确性（test1，以未舍入 `Clinic` 做指纹） | ✅ 339/339 唯一匹配且 ID 一致 |
| test1 / test2 的 `Clinic`、`Combined` 与 heatmap 的 Pearson | ✅ **1.0000 / 1.0000** |

### ⚠️ 三条使用警告（务必遵守）

| # | 警告 | 证据 |
|---|---|---|
| **W1** | **绝不能按行位次配对任何两张表** | 各文件行序互不相同（`heatmap_data_all` 按 `Combined` **升序**，分数文件另一套顺序，`_cases_with_ids` 按 `patient_index`）。按位次配对的指纹检验：test1 仅 **36/339** 行精确、test2 仅 **14/189** 行；**按 ID 则 339/339、189/189 精确** |
| **W2** | **`_cases_with_ids.csv` 的 `real_id` 对训练队列不可靠** | 三个外部队列指纹验证 **完全正确**（test1 339/339、test2 189/189，Pearson **1.0000**）；训练队列同一检验 **0/398**，Pearson 0.091。→ **训练队列请用 `heatmap_data_all.csv` 或 `_fullprec_scores.csv`**（两者在 1016 例上逐位相同） |
| ~~W3~~ | ~~`heatmap_data_all` 的生境列是另一模型~~ | ❌ **此条已作废（2026-10-16 更正）**。`heatmap_data_all.csv` 与 `_fullprec_scores.csv` 在**全部 1016 例、全部列**上逐位相同；与部署分数在 test1（339/339）、test2（189/189）完全相同。先前看到的"不同"完全是 **W2 的训练队列映射错误**造成的假象。**只有一个部署模型。** |

### 复算入口

```
python verify/check_id_linkage.py     # A 结构 / B 覆盖 / C 指纹 / D 位次对照 / W3 / W4
→ results/tables/id_linkage_checks.csv
```

### 实现

新代码库把它封在 [`src/hrpca/linkage.py`](../src/hrpca/linkage.py)：
`load_cases_with_ids` / `attach_identifiers` / `load_cohort` / `load_analysis_frame` /
`verify_key`（自检报告）。装配脚本 `src/00_assemble_data.py` 已把
`_cases_with_ids.csv` → `data/cases_with_ids.csv`、`_fullprec_scores.csv` →
`data/fullprec_scores.csv` 纳入 `MANIFEST.json`。

### 用正确连接后的修正结果

| 原先的错误结论 | 正确结论 |
|---|---|
| 「Table 1 有 45 处不符」 ✗ | **Table 1 分类变量 28 格中 27 格完全正确**；唯一差异是训练队列 ISUP grade 3（76 vs 83） |
| 「部署 Clinic 不是 ISUP+PI-RADS 的普通 logistic」 ✗ | **是**；test1/test2 概率相关 **1.0000**，最大差 0.0031 |

---

## 四、本仓库（新代码库）的版本纪律

1. **唯一数据入口** = `data/`（由 `python src/00_assemble_data.py` 从部署源装配，附 SHA256 清单）。
2. **禁止读取** 任何名字含 `副本`、`_deprecated`、`_deleted`、`回退镜像`、`xxx\` 的路径。
3. 高精度分数**只用** `heatmap_data_all.csv`；Combined 一律用 `derived_data\*_scores_label.csv`。
4. 论文表格 AUC 一律 `round(x, 3)`；`0.863` 属两位小数舍入伪影，**全库已统一为 0.864**。
5. **跨表连接只允许经 `_cases_with_ids.csv`（或 `heatmap_data_all.csv` 的 ID）**，
   **严禁按行位次配对**（见 W1）；训练队列的 ID 用 `heatmap_data_all.csv`（见 W2）。
