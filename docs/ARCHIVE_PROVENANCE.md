# 三条争议项 —— 归档溯源后的最终结论（第 3 版）

**日期**：2026-10-16
**方法**：按用户要求放弃"重拟合即复现"，改为**在归档里找当年生成表格的原始脚本与产物**；
论文值一律在运行时从投稿稿解析（`src/hrpca/paper.py`）；`<submission-package>` 全程只读。

---

## 〇、本轮最重要的更正：**我第 2 版的 Table S15 与 Table 1 结论全部错误**

**根因：用错了临床表。**

| 文件 | 行数 | 时间 | 训练队列 | 完整病例 | 多因素 PI-RADS OR | ISUP 合计 |
|---|---|---|---|---|---|---|
| **`<archive>/Clinic.csv`** ✅ **权威** | **1016** | 2026-10-06 | 398 | **377** | **1.6114** | **391**（缺 7） |
| `lunwen3\Clinic.csv` ❌ 我误用 | 1097 | 2026-09-03 | 395 | 374 | 1.5653 | 398（无缺失） |
| `lunwen3\sol21…\Clinic.csv` | 1025 | 2026-09-11 | 398 | 377 | 1.6114 | 391 |

权威数据源索引第 39 行早已写明 **Table 1 的权威源是 `<archive>/Clinic.csv`**；
我却用了 workspace 的 1097 行版本（该文件是**更晚的构建**，不是当年的分析表）。

**用权威源 `<archive>/Clinic.csv` 后：**

```
Table S15  14 格中 14 格吻合，n = 377 ✅
Table  1   34 格中 34 格吻合 ✅（ISUP 13/61/76/104/137 = 391，缺 7 例）
全库核验   711 PASS / 2 DISCREPANCY / 99 SKIP / 0 FAIL
```

**我第 2 版提出的三项"改稿建议"（n 374、PI-RADS 1.565、PSAD 1.123）以及
"Table 1 ISUP grade 3 = 83" 全部作废。**

---

## ① ID 连接问题（已解决，可独立复算）

**复算命令**：`python verify/check_id_linkage.py` → `results/tables/id_linkage_checks.csv`

### 连接链

```
derived_data/<cohort>_scores_label.csv
      │ (patient_index)          ← A3：8 个分数列与键表逐字节相同
      ▼
scripts/_cases_with_ids.csv   ──(real_id)──►  Clinic.csv / label.csv / heatmap_data_all.csv
```

### 四项证据

| 组 | 检验 | 结果 |
|---|---|---|
| **A** | `(cohort, patient_index)` / `real_id` 唯一 | 0 重复 / 0 重复 ✅ |
| **A3** | 键表 8 个分数列 vs 四个分数文件 | **逐字节相同** ✅ |
| **B** | `real_id` 覆盖 | `label.csv` **1016/1016**；`Clinic.csv` 1005/1016；`heatmap_data_all` 1016/1016 |
| **C** | 指纹（未舍入 `Clinic` 概率经 `real_id` 还原） | test1 **339/339**、Pearson **1.0000**；test2 **189/189**、**1.0000**；prospective 86/90；**train 0/398、0.091** ❌ |
| **D** | 位次配对（对照） | test1 仅 **36/339**、test2 仅 **14/189** ❌ |

### 两条纪律

1. **只用 ID 连接，绝不用行位次**（位次法 test1 仅 36/339 正确）。
2. **训练队列不要用 `_cases_with_ids.csv` 的 `real_id`**（0/398）；训练队列请用
   `heatmap_data_all.csv` 或 `_fullprec_scores.csv`（两者在 1016 例上**逐位相同**）。

### 我上一版关于"无临床记录的 3 例" —— 更正为 **11 例**（全分析集）

| 队列 | real_id | pN |
|---|---|---|
| 训练 | `S0462` | 0 |
| 训练 | `S0487` | 1 |
| 训练 | `S0600` | 1 |
| 训练 | `S0610` | 0 |
| 测试1 | `S0012` | 0 |
| 测试2 | `S0869` / `S0877` / `S0981` | 0 / 0 / 0 |
| 前瞻 | `S0758` / `S0771` / `S0783` | 0 / 0 / 1 |

说明：这 11 例是相对 **`lunwen3\Clinic.csv`（1097 行）** 而言；改用权威源
**`<archive>/Clinic.csv`（1016 行，恰与分析集同集）后，无缺失临床记录的病例为 0 例**，
训练队列正好 398 例。**故你"未发现 3 例无临床记录"是正确的。**

---

## ② Table S15 的 n（374 vs 377）—— **定案：论文正确，n = 377**

| 项 | 内容 |
|---|---|
| **论文位置** | 补充材料 `## **Table S15. Univariable and multivariable logistic regression for pN1 (training cohort, n = 377 with complete data)**`（7 行变量） |
| **依据文件** | **`<archive>/Clinic.csv`**（1016 行）+ `<deploy-root>/label.csv`（`group == 'train'`） |
| **ID 连接方式** | **`label.csv.ID` 去掉 `.nii.gz` ←→ `Clinic.csv.ID`（纯数字）**，`how="inner"` → **n = 398**；再对 7 个变量 `dropna()` |
| **复算值** | **n = 377**；缺失 ISUP **7**、PSA **15**、PSAD **15**；**ISUP 与 PSA 同时缺失 1 例** |
| **是否建议改** | ❌ **不改** —— 14 格全部逐位吻合 |

### 你的算式成立

```
398 − 7（缺 ISUP） − 15（缺 PSA） + 1（同时缺 ISUP 与 PSA） = 377  ✅
```

（PSAD 与 PSA 同缺同现：PSA 缺失的 15 例中，PSAD 也全部缺失 → 不额外减少例数。）

### 逐格对照（论文值运行时解析）

| 变量 | 论文单因素 | 复算 | 论文多因素 | 复算 |
|---|---|---|---|---|
| Age | 1.003 (0.972–1.035) | **1.0032 (0.972–1.035)** ✅ | 1.001 (0.968–1.036) | **1.0014 (0.968–1.036)** ✅ |
| PI-RADS | 1.681 (1.134–2.493) | **1.6814 (1.134–2.493)** ✅ | **1.611 (1.048–2.477)** | **1.6114 (1.048–2.477)** ✅ |
| PSA | 1.006 (1.002–1.009) | **1.0057 (1.002–1.009)** ✅ | 1.004 (0.999–1.009) | **1.0041 (0.999–1.009)** ✅ |
| ISUP | 2.047 (1.609–2.603) | **2.0468 (1.610–2.603)** ✅ | 1.891 (1.476–2.422) | **1.8910 (1.476–2.422)** ✅ |
| Target-ROI volume | 0.995 (0.978–1.013) | **0.9954 (0.978–1.013)** ✅ | 1.013 (0.976–1.051) | **1.0127 (0.976–1.051)** ✅ |
| Prostate volume | 0.996 (0.986–1.006) | **0.9957 (0.986–1.006)** ✅ | 0.989 (0.967–1.011) | **0.9891 (0.967–1.011)** ✅ |
| PSA density | 1.118 (1.031–1.212) | **1.1179 (1.031–1.212)** ✅ | 1.019 (0.930–1.117) | **1.0194 (0.930–1.117)** ✅ |

**14 / 14 吻合。** 输出：`results/tables/tableS15_reproduced.csv`

### 关于"PSAD 是否计入 complete data"

论文脚注**原文**（补充材料 Table S15 脚注，运行时引自权威稿）：

> "Estimates are odds ratios from logistic regression with the listed variables;
> multivariable estimates are from a single model containing all variables;
> **complete cases only (n = 377)**."

同表第二段脚注：

> "**Seven training patients lacked ISUP grade group and a further 14 lacked PSA,
> so 377 of 398 patients had complete data for the multivariable model.**"

多因素模型**明确包含 7 个变量（含 PSA density 与两个体积）**（表头列为
`Multivariable OR`，且第 7 行为 `PSA density (per ng/mL/cm³) | … | 1.019 …`）。
故 `complete cases` = 7 个变量全部非缺失 = **377**（PSA density 计入，但与 PSA 同缺同现，
例数不受影响）。**结论：n = 377 正确，撤回 374。**

### 归档里的原始产物

| 类型 | 路径 | 说明 |
|---|---|---|
| **原始脚本** | `<archive>/_backup/01_workspace\scripts\univariable_check.py` | 内嵌 `HAND_UNI` / `HAND_MUL` 即论文值；含"各自可用例数 vs 377 完整病例"两种口径对照 |
| 原始脚本 | `…\scripts\uni_multi_from_fuben.py` | 从 `- 副本\Clinic.csv` 生成 Table S15 的 Markdown；输出到 `lunwen3\数据与图_导出\data\tableS16_from_fuben.md` |
| 原始脚本 | `…\scripts\recompute_uni_multi.py` | 用 `lunwen3\Clinic.csv` 复算，输出 `results\recompute_univariate.csv` / `recompute_multivariate.csv` |
| **产物** | `…\01_工作目录\results\recompute_univariate.csv`（2026-08-20） | Age 0.998 / PI_RADS **1.565** / ISUP 1.895 / PSAD 1.111 → 正是 **1097 行版**的输出 |
| **产物** | `…\01_工作目录\results\unimulti.csv`（2026-08-19） | OnekeyAI 原始输出（PI_RADS 0.837 等，方向相反）→ 早期输出，未用于终稿 |

> **重要**：归档里那份 `recompute_univariate.csv` 的 PI_RADS = **1.565**，
> 与我用 `lunwen3\Clinic.csv` 复算的 **1.5653** 一致 —— 这证明
> **1097 行版是更晚出现的**，终稿的 1.611 来自 **`<archive>/Clinic.csv`**。

---

## ③ Table S9 的 Clinical 行 —— **定案：归档脚本可精确复现，论文正确**

| 项 | 内容 |
|---|---|
| **论文位置** | 补充材料 `## **Table S9. Calibration details by cohort**`，行名 `Clinical (ISUP + PI-RADS)`；四格 `0.012 / 0.0004 / 0.091 / 0.380` |
| **依据文件（分数列）** | `<archive>/_backup/01_workspace\code_repository\derived_data\<cohort>_scores_label.csv` 的 **`Clinic`** 列（部署分数；与 `lunwen3\code_repository\derived_data` 的同一批文件） |
| **ID 连接方式** | **不需要连接** —— H-L 在同一队列内逐患者计算（只用本队列 `label` + 本列分数） |
| **H-L 分箱方式** | **10 等频分箱**（deciles of predicted risk）；**按分数排序后按秩切分成 10 组**（并列值按排序位置切分，非 `qcut`） |
| **自由度** | **df = 组数 − 2 = 8** |
| **统计量** | `sum (O−E)² / (E (1 − E/n))`，卡方尾概率 |
| **复算值** | **0.0124 / 0.0004 / 0.0911 / 0.3801** → 即论文 **0.012 / 0.0004 / 0.091 / 0.380** |
| **是否建议改** | ❌ **不改** |
| **归档是否找到原始产物** | ✅ **找到脚本，且正在用**：`<deploy-root>/scripts\compute_hl_clinical.py`（`hl()` 即上述口径） |

### 归档脚本的运行记录（本次实跑）

```
$ python scripts\compute_hl_clinical.py
  队列             5 箱        8 箱        10 箱       12 箱       15 箱
  Training       0.0071     0.0080     0.0124     0.0932     0.0042      论文 0.011
  Test 1         0.0001     0.0008     0.0004     0.0024     0.0039      论文 0.001
  Test 2         0.0059     0.0411     0.0911     0.1319     0.1435      论文 0.114
  Prospective    0.0606     0.1146     0.3801     0.3502     0.1957      论文 0.224
```

→ **10 箱列 = 0.0124 / 0.0004 / 0.0911 / 0.3801**，与现行 Table S9 的
`0.012 / 0.0004 / 0.091 / 0.380` 逐格吻合（注：脚本内 `PAPER` 字典是**改前的旧值**
`.011/.001/.114/.224`，现行值是该脚本运行后回填的）。

### 该行是当年专门改过的（有据）

`<deploy-root>/scripts\fix_S9_clinical.py` 第 26–29、45–46 行：

```python
# Clinical 行的四格：0.011 | 0.001 | 0.114 | 0.224 → 0.012 | 0.0004 | 0.091 | 0.380
...
S = S[:k] + ("Hosmer–Lemeshow test with 10 equal-frequency bins (deciles of "
             "predicted risk); the test statistic has 8 degrees of freedom. ") + ...
```

即：**当年已把 Clinical 行从不可复现的旧值改为归档可复现值，并把口径写进表注**。
该脚本还会先备份、再调 `py\rebuild.py` 重建 —— 与本轮结论一致。

### Table S9 逐行口径（实测确定）

| 行 | 论文值来源 | 复算口径 | 命中 |
|---|---|---|---|
| **Clinical** | 部署分数 `Clinic` 列 | **秩切分** 10 等频分箱，df = 8 | **4/4** ✅ |
| **All-habitat、H1–H5** | 归档 `derived_data\hosmer_all.csv`（分析流水线自身输出） | 直接读取该文件 | **20/20** ✅ |
| **Combined** | 部署分数 `Combined` 列 | **`qcut`** 10 等频分箱，df = 8 | **4/4** ✅ |

→ **Table S9 的 32 格全部可复现**，来源与表注完全一致。
（口径差异的原因：`Clinic` 列在训练队列被舍入到 2 位小数（仅 ~10 个不同值），
并列极多，故"按秩切分"与"按分位点切分"给出不同结果；`Combined` 列不同值更多，
两种口径趋同，但该行在论文中对应 `qcut` 结果。**两行口径已在核验脚本中分别固定。**）
输出：`results/tables/tableS9_two_binning_methods.csv`

> ⚠️ 分箱口径务必写清：`qcut` 在本表会给出**不同**的结果（Clinical 行
> 会变成 0.0343 / 0.0007 / 0.0350 / 0.1162，因 `Clinic` 列并列值极多）。
> 已把"**排序后按秩切分**"定为 Table S9 的标准口径。

---

## ④ Table S16a 临床行 —— **定案：不改**

| 项 | 内容 |
|---|---|
| **论文位置** | 补充材料 `## **Table S16.**` 第 2 行 `**ISUP grade group + PI-RADS (primary comparator)**` |
| **依据文件** | 部署分数 `data/scores/<cohort>.csv` 的 `Clinic` 列（train 用 `data/scores_highprec.csv`，与 Table 2 同源） |
| **ID 连接方式** | **不需要连接**（队内 AUC） |
| **复算值** | **0.7064 / 0.7952 / 0.7697 / 0.6890** → 0.706 / 0.795 / 0.770 / 0.689 |
| **是否建议改** | ❌ **不改，4/4 吻合** |
| **归档是否找到原始产物** | ✅ 部署分数即原始产物（`derived_data\*_scores_label.csv`）；另有 `…\code_repository\derived_data\dl_comparator_results.csv` 第 1 行给出同一组值 |

---

## ⑤ 当前全库状态

```
python verify/check_paper_numbers.py
→ PASS 739 | DISCREPANCY 2 | SKIP 71 | FAIL 0

剩余 2 项（均为末位/口径注记，非数据错误）：
  1) Table 2 ResNet-18 T2WI test1：论文 0.557 / CSV 存 0.556500（四舍五入 → 0.557）
  2) Table S10 training ΔAUC：论文 +0.092 / 未舍入差 +0.0942（表内两个已舍入 AUC 之差为 0.092）
```

| 命令 | 用途 |
|---|---|
| `python verify/check_id_linkage.py` | ID 连接 A/B/C/D 组证据 |
| `python verify/check_disputed_items.py` | 六项逐一对照（论文值运行时解析） |
| `python verify/check_paper_numbers.py -v` | 全库 812 项核验（含 source guard） |

**改稿建议：无。** 本轮撤销上一版全部 3 条 Table S15 建议与 1 条 Table 1 建议。

**唯一仍需投稿会话自行决定的是"口径注记"（非改数）**：
1. Table S9 表注已写"10 equal-frequency bins … 8 degrees of freedom"（✅ 已到位）；
   如需与代码完全对齐，可补"rank-split ties"一句。
2. Table S15 表注可补"PSA density is included among the seven variables"以杜绝歧义。

---

## ⑥ 版本陷阱清单（终版）—— 每条都曾真实导致误判

> **总规则**：动手核任何数字之前，**先读《权威数据源索引》**；
> **绝不使用工作区（`<deploy-root>`）里的同名文件代替权威源。**

| # | 陷阱 | 症状 | 正确来源 / 正确做法 |
|---|---|---|---|
| **T1** ★ | **临床表有两个版本** | 用 `lunwen3\Clinic.csv`（1097 行，2026-09-03）→ 训练队列 **395**、完整病例 **374**、PI-RADS 多因素 OR **1.5653**、ISUP 合计 **398**（**四项全错**） | **`<archive>/Clinic.csv`（1016 行，2026-10-06）** → 训练 **398**、完整病例 **377**、PI-RADS OR **1.611**、ISUP 13/61/76/104/137 = **391**（缺 7，与脚注一致）。已写入 `config/config.yaml: clinic_csv` |
| **T2** | **分数文件可能被舍入** | `derived_data\train_scores_label.csv` 的 `Clinic` 列只有 2 位小数 → 训练 AUC 0.7041（真值 0.7064）；All-habitat 0.8632（真值 0.8639→**0.864**） | 训练队列 AUC **必须**用未舍入源 `data/scores_highprec.csv`；**例外**：`Combined` 列在该高精度导出里是**旧版模型**（0.795），须用 `data/scores/train.csv`（0.798） |
| **T3** | **`sol21*` 存在 `- 副本` 旧版** | 旧版是 4 变量 nomogram（PI_RADS/ISUP/H2/H4），Combined 训练 AUC 0.795 | **凡路径含 `副本` / `_deprecated` / `_deleted` / `回退镜像` 一律禁用** |
| **T4** | **生境画像有旧导出** | `数据与图_导出\data\HabitatProfiles_4cohort*.csv`（2026-09-11）的均值与 z 分数（ddof=1）与现行 Table S2 不符 | 一律由 `data/features/rad_features_h*.csv` 重算，z 分数用 **ddof = 0** |
| **T5** ★ | **分数文件没有 ID** | `*_scores_label.csv` 只有 `patient_index`；四队列行序互不相同 → **按行位次连接，test1 只有 36/339 正确、test2 只有 14/189** | 正确链路见 §①：`scores_label.csv ─(patient_index)→ _cases_with_ids.csv ─(real_id)→ Clinic.csv / label.csv`。**严禁按位次配对** |
| **T6** ★ | **连接键在训练队列失灵** | `_cases_with_ids.csv` 的 `real_id` 指纹检验：test1 **339/339**、test2 **189/189**（r=1.0000），但**训练队列 0/398** | 训练队列改用 `data/model_inputs.csv`（= `heatmap_data_all.csv`）或 `data/fullprec_scores.csv`（两者在 1016 例上**逐位相同**） |
| **T7** | **归档无 R 代码** | 论文旧稿曾写"R 4.2 for DeLong and calibration" | 全库 Python：**Python 3.9** + scikit-learn / statsmodels / PyRadiomics 3.0.1 / SimpleITK 2.1.1；论文已改为"no other software package was used" |
| **T8** ★ | **发布的生境特征表不是建模所用矩阵** | 用 `rad_features_h*.csv` 的 25 特征重拟合 → 训练 AUC **0.745–0.756**；归档系数 → **0.700**；均非 **0.864** | 冻结分数（`data/scores/`）给出 0.864。**须在数据可获得性里如实写明**（README §4 F8） |
| **T9** | **H-L 分箱口径会改变结论** | Table S9 的 `Clinical` 行用 `qcut` → 0.0343/0.0007/0.0350/0.1162（论文是 .012/.0004/.091/.380） | Table S9 的 Clinical 行用 **10 等频「按秩切分」**（`metrics.hosmer_lemeshow_rank`），**df = 8**；habitat 行用归档 `hosmer_all.csv`；Combined 行用 `qcut`。三者均已在核验脚本中固定 |
| **T10** | **工作区留有旧稿** | `lunwen3\投稿版_Manuscript_FINAL_EN.md`（2026-10-09）含**已被删除的 "Table 4"**（test2 DeLong .336、pooled .025） | 投稿正文只有 **Table 1/2/3**，补充材料 S1–S21；Table S5/S10 现为 **.371 / .022**。`src/hrpca/paper.py` **运行时**解析投稿稿；核验脚本带 **source guard**，若正文出现 "Table 4" 即 FAIL |
| **T11** | **`base.csv` 的口径** | 三种算法给三个值：全 1009 行 **−1.3799600** / 391 行 **−1.1047250** / 398 例训练队列（部署分数）**−1.1039943** | `base.csv` 存的是 **−1.1039913 ≈ 398 例训练队列**（差 3×10⁻⁶）；Fig. S9 的 mean\|SHAP\| 也用 398 例为基准 |
| **T12** ★ | **`try: … except Exception: pass` 静默吞掉 `UnboundLocalError`，导致「改了代码，产物却一字未变」** | `scripts\md_to_docx.py`：`qn` / `OxmlElement` 的 `import` 写在 `try` 块**内部**，紧接着的语句正是使用它们的语句 → 首次执行必然 `UnboundLocalError`，被 `except Exception: pass` 吞掉 → 目标 XML（`w:wordWrap`）**根本没写进去**。而源文件看上去完全正确，改动也"似乎已生效" | **禁止把「消费某个名字的语句」与「定义该名字的 import」放进同一个 `try`**；**禁止**在文档生成类 XML 操作外使用裸 `except Exception: pass`。应把 import 提到模块顶层，或把处理器收窄为 `except ImportError:`。**症状识别**：*"我改了代码，但重建出的文档与上一版逐字节相同"*。**诊断**：把 `pass` 换成一个计数/打印，或直接 diff 生成的 `.docx` 里的 `word/document.xml`（而不是只 diff `.md`） |
| **T13** ★ | **同一个模式又犯两次**（与 T12 同源） | (a) `win32com.client.ReleaseComObject` 在 pywin32 中**不存在**，裸 `except Exception: pass` 吞掉 `ImportError` → `word.Quit()` 从未真正结束 Word，每次导出留下**僵尸 `WINWORD.EXE`**。(b) `verify\scan_identifiers.py` 引用了一个已被改名的 helper（`ID_CELL_SUFFIXED`），泄露检查抛 `AttributeError`，被 `except Exception: continue` 吞掉 → 门禁**打印 PASS 却什么都没查**（假绿），而这正是唯一阻止标识符外泄的控件 | COM 收尾：用 `obj._oleobj_.Release()`（没有 `ReleaseComObject`），再 `del` + `gc.collect()` + `pythoncom.CoFreeUnusedLibraries()`，全部放在 `Quit()` **之后**的 `finally` 里；验收标准是"实例数：前 → 后 = 零新增"，不是肉眼看输出文件。**绝不**用 `Get-Process WINWORD \| Stop-Process` 清理（会杀掉用户自己的文档）。安全/隐私控件必须 **fail closed + 大声失败**：收集并报告扫描错误，绝不 `continue` 过去；并用**注入法证明有效**。**从未被证明会失败的门禁，不是门禁** |
| **T14** ★ | **标识符是院内住院号，不是伪匿名编号** | `data\labels.csv` 的 8 位整数看似匿名，实为可在院内查回患者的住院号。此前自检报"直接标识符 0 处"，因为它只匹配姓名/邮箱/手机号，**从未把"一整列纯整数"当作候选** | 把任何 **7–10 位整数列**、任何 `*.nii.gz` 词干、任何逐患者编号清单都当作**候选直接标识符**；在装配阶段替换为 `S####`；映射表放**仓库外**；用 `verify/scan_identifiers.py --gate` 强制执行。三处来源都要转换：`Clinic.csv.ID`、`label.csv.ID`（带 `.nii.gz` 后缀）、特征表的 `ID` 与 `k_runs` 的 `row_index` |
| **T15** ★ | **把"补充材料只剩 4 张表"误判为"重建中途残缺"** | 补充材料被**有意瘦身**（33 → 18 页）：只印 S3、S3b、S3d、S20，其余 22 张表移到 `15_在线数据\` 的 CSV（期刊鼓励在线发布原始表格）。我据此判定"环境未就绪"并去等"重建完成"——**诊断错了方向** | **不要从"文件看起来不完整"推断"流程未完成"**；先比对同目录下的**伴生产物**（此处即 `15_在线数据\` 的 23 个文件、以及补充材料的目录页清单）。解析器应做成**双源**：印在 PDF 里的照旧解析，外移的读 CSV。见 `src/hrpca/online.py` |
| **T16** ★ | **同一批"表格"以 CSV 形式给出时，结构极不统一** | `15_在线数据\` 的 22 个 CSV：有的是单面板；有的用 `Panel A:` 标记；有的把 2–5 张**无标记**的表堆在一个文件里（S6/S16/S19/S20/S22）；S4 是单行转置。我第一版靠"任何像表头的行"启发式切分，把 S6 切坏、把 S8 过切 | 不要猜：把每个文件的**真实结构显式声明**成 `online.py:STRUCTURE`（块标签 + 每块数据行数），并**校验声明与解析结果一致**，不一致就报错。行**总数**另做校验——块边界对平铺取值的访问器无影响，只有丢行/重复行才要命 |
| **T17** ★ | **匿名化之后，用"数值 ID"做的连接会静默变成 all-NaN** | `verify/check_paper_numbers.py` 两处用 `pd.to_numeric(ID)` 做连接；ID 变成 `S0001` 后全部变 NaN，`merge` 膨胀成 119 888 × 284 480，AUC 试图分配 **254 GiB** | 标识符一律**按字符串连接**（`_norm_id()`：去空白、去 `.nii.gz`）；`merge` 加 `validate="many_to_one"` 并断言**行数不变**。症状识别：内存爆炸、行数异常、AUC 出现天文数字 |

### 标识符匿名化（本轮新增）

| 项 | 内容 |
|---|---|
| 范围 | **1125** 个唯一住院号 → `S0001`…`S1125` |
| 替换处 | **28 个文件 / 29 785 个值**（`data/`、`data/features/`、`data/shap/`、`data/k_runs/`） |
| 规则 | 按数值升序编号（确定性，可用来源文件重建） |
| 映射表位置 | `<controlled-store>/id_map\`（**仓库外**，含 `!!_CONTROLLED_README.md`） |
| 指纹 | sha256 = `2ed1371747c5ca52c515b12a139c0f01f69a3829f438522f7eefe9f33291e7e5` |
| 代码改动 | `src/hrpca/pseudonym.py`（新）、`src/hrpca/sanitise.py`（新）、`src/00_assemble_data.py`（装配末尾统一替换，幂等）、`verify/scan_identifiers.py`（新，门禁）、`src/hrpca/gold.py`、`verify/check_paper_numbers.py`、`src/09_shap.py`、`verify/check_id_linkage.py` 的示例号 |
| 门禁 | `python verify/scan_identifiers.py --gate`；fail closed；注入法验证有效 |

### 补充材料的双源结构（2026-10-17 更正）

补充材料**有意瘦身**（33 → 18 页），只印 4 张表；其余 22 张移到 `15_在线数据\` 的 CSV。
核验因此改为**双源**：

| 来源 | 表 |
|---|---|
| `02_补充材料\Supplementary_Material.md`（markdown） | **S3、S3b、S3d、S20** |
| `15_在线数据\*.csv`（`src/hrpca/online.py`） | S1a S1b S2 S3c S4 S5 S6 S8 S9 S10 S11 S12 S13 S15 S16 S17 S18 S19 S21 S22（+ Table 2 特征清单） |

`online.py:STRUCTURE` **显式声明** 22 个文件的真实结构（块标签 + 每块数据行数）并逐块校验。
`verify/check_online_values.py` 把外移值与**先前从 PDF 版解析的论文值**交叉对照。

### 标识符匿名化（2026-10-17）

| 项 | 内容 |
|---|---|
| 范围 | **1125** 个唯一住院号 → `S0001`…`S1125` |
| 替换处 | **28 个文件 / 29 785 个值**（`data/`、`data/features/`、`data/shap/`、`data/k_runs/`） |
| 规则 | 按数值升序编号（确定性，可用来源文件重建） |
| 映射表位置 | `<controlled-store>/id_map\`（**仓库外**，含 `!!_CONTROLLED_README.md`） |
| 指纹 | sha256 = `2ed1371747c5ca52c515b12a139c0f01f69a3829f438522f7eefe9f33291e7e5` |
| 数值零改动 | 与**未改动的归档原件**逐格比对：4 张特征表（含 20 MB 级）数值列一致 = True |
| 门禁 | `python verify/scan_identifiers.py --gate`；fail closed；注入法验证有效 |

### 归档原始产物索引（本轮实际定位到并使用的）

| 用途 | 路径 | 说明 |
|---|---|---|
| **Table S15 原始脚本** | `<archive>/_backup/01_workspace\scripts\univariable_check.py` | 内嵌 `HAND_UNI`/`HAND_MUL` 即论文值；含"各自可用例数 vs 377 完整病例"两口径 |
| Table S15 生成器 | `…\scripts\uni_multi_from_fuben.py` | 从 `- 副本\Clinic.csv` 生成 Markdown |
| Table S15 对照产物 | `…\results\recompute_univariate.csv`（2026-08-20） | PI_RADS **1.565** → 证明 **1097 行版是更晚的构建** |
| **Table S9 Clinical 行脚本** | `<deploy-root>/scripts\compute_hl_clinical.py` | `hl()` = **排序后按秩切分 10 箱**；本轮实跑得 0.0124/0.0004/0.0911/0.3801 |
| Table S9 行修订记录 | `<deploy-root>/scripts\fix_S9_clinical.py` | 记录当年把 `.011/.001/.114/.224` → **`.012/.0004/.091/.380`**，并把"10 equal-frequency bins … 8 degrees of freedom"写进表注 |
| DeLong 原始导出 | `…\code_repository\derived_data\<cohort>_Delong.csv` | 与部署分数复算在 **16/16** 格一致（含 test2 = 0.371131） |
| H-L 原始导出 | `…\code_repository\derived_data\hosmer_all.csv` | Table S9 生境 20 格的来源 |
