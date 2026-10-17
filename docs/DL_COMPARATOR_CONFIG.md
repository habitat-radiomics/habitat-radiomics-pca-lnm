# Deep-learning comparator — configuration and provenance

Everything on this page is **read from the authors' archive**; nothing here is a
reconstruction.  No imaging data, model weights or patient-level files are
copied into this repository — only the configuration facts needed for a reader
to rebuild the comparator.

| Item | Value |
|---|---|
| Archived project | `C:\O\深度学习建模\sol12_DL_T2WI_18` (T2WI variant; sibling projects exist for ADC and DWI) |
| Authoring notebook | `Step1. 3D深度学习模型.ipynb`, cell 3 (`2026-10-06 09:47:42`) |
| Training driver (alternative) | `run_cv10_train.py` (`2026-10-01 03:43:26`) — **not** what produced the archived model, see §5 |
| Run configuration | `config.yaml` (`2026-10-06 10:50:42`) |
| Saved model | `dl_models\resnet18\viz\BEST-training-params.pth` (`2026-10-06 09:43:50`) |
| Saved run metadata | `dl_models\resnet18\viz\task.json` |
| Framework | OneKeyAI `onekey_algo.classification3d.run_classification3d.main` (3D branch) |
| Paper text it backs | Supplementary Methods **S4** (deep-learning comparator) |

---

## 1 · Architecture — ResNet-18 (3D), **not** a Vision Transformer

The architecture is unambiguous; four independent sources agree.

**(a) `config.yaml`, lines 27–28**

```yaml
model_names: [resnet18]
sel_dl_model_name: resnet18
```

Line 32 sets the schedule:

```yaml
epoch: 50
```

**(b) `dl_models\resnet18\viz\task.json`** (whole file)

```json
{
 "model_name": "resnet18",
 "num_classes": 2,
 "squeeze_dim": null,
 "in_channels": 1,
 "input_size": [48, 48, 32],
 "vit_settings": null
}
```

`vit_settings` is **null**: the ViT branch of the framework was not used.

**(c) The module summary printed by the pipeline itself** —
`Step2. 3D特征提取.ipynb`, lines 1102–1180:

```
[2026-10-06 09:47:30 - ...comp2>: 143] INFO 成功加载dl_models\resnet18\viz\BEST-training-params.pth模型参数。
Feature name:  || Module: ResNet(
  (conv1): Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
  (bn1): BatchNorm3d(64, ...)
  (maxpool): MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (0): ResNetBlock( conv1: Conv3d(64, 64, 3x3x3) ... conv2: Conv3d(64, 64, 3x3x3) ...)
    (1): ResNetBlock( ... )
  ...  layer2: ResNetBlock(Conv3d(64, 128, stride=(2,2,2))) ...
  ...  layer3: ResNetBlock(Conv3d(128, 256, stride=(2,2,2))) ...
  ...  layer4: ResNetBlock(Conv3d(256, 512, stride=(2,2,2))) ...
  (avgpool): AdaptiveAvgPool3d(output_size=(1, 1, 1))
  (fc): Linear(in_features=512, out_features=2, bias=True)
)
```

**(d) The saved weights** (`BEST-training-params.pth`, load with
`torch.load(..., map_location="cpu")["state_dict"]`):

| Property | Value |
|---|---|
| Parameter tensors | **104** |
| Total parameters | **32 995 027** (≈ 3.3 × 10⁷ — the figure quoted in Methods S4) |
| Key pattern | `conv1 · bn1 · layer1..4 · fc`; **no** `patch_emb`, `transformer` or `pos_emb` key |
| Channel ladder | 64 → 64 → 128 → 256 → 512 (2 blocks per stage) |
| Stem | `conv1.weight` = (64, 1, 7, 7, 7) — single input channel, 3-D kernel |
| Classifier | `fc.weight` = (2, 512) |
| Shortcuts | no `downsample` keys: residual blocks use an **identity** shortcut and apply the resolution change in the first convolution of the block |

> **Do not describe this as MONAI's `resnet18`.** The framework's own 3-D
> ResNet-18 differs from `monai.networks.nets.resnet18`: the MONAI model has 125
> parameter tensors / 33 171 542 parameters because it adds a 1 × 1 convolution
> to the three down-sampling shortcuts.  The archived model has neither, so the
> two are **not** the same object.  Refer to it as *“a 3-D ResNet-18 with
> identity-shortcut basic blocks”* (OneKeyAI implementation).
>
> The stem's `conv1` uses `stride=(1, 2, 2)` — geometric consistency only, since
> the input is cubic 48 × 48 × 32.

## 2 · Training configuration actually used for the archived model

From `Step1. 3D深度学习模型.ipynb`, cell 3 (verbatim, abridged to the parameters):

```python
roi_size = [48,48,32]
normalize_method='imagenet',
model_name=model_name,          # from config.yaml  ->  resnet18
roi_size=roi_size,
batch_size=16,
epochs=get_param_in_cwd('epoch'),   # config.yaml line 32  ->  50
init_lr=1e-5,
optimizer='adam',
val_interval=1,
save_per_epoch=True,
pretrained=True,
```

| Hyper-parameter | Value | Source |
|---|---|---|
| Architecture | 3-D ResNet-18 (identity-shortcut basic blocks) | `config.yaml:27-28`, `task.json` |
| Input channels | 1 | `task.json` |
| Input size (voxels, D×H×W order as stored) | 48 × 48 × 32 | `task.json`, notebook cell 3 |
| Normalisation | `imagenet` | notebook cell 3 |
| Pre-training | `pretrained=True` | notebook cell 3 |
| Optimiser | Adam | notebook cell 3 |
| Initial learning rate | 1 × 10⁻⁵ | notebook cell 3 |
| Batch size | 16 | notebook cell 3 |
| Epochs | 50 | `config.yaml:32` |
| Validation interval | every epoch | notebook cell 3 |
| Checkpoint policy | `save_per_epoch=True`, best-validating epoch retained | notebook cell 3 |
| Output features | 512-d global-average-pooled activations | model summary `(avgpool)` + `(fc) Linear(512 → 2)` |
| Data split | `split_info\train-RND-0.txt` (358 cases) / `val-RND-0.txt` (40 cases) | `config.yaml:10-11` |
| Random seed / folds | single split (`-0`); no cross-validation in the archived run | see §5 |

## 3 · Where the ViT parameters come from (benign, unused)

Cell 3 also defines

```python
vit_settings = {
        'image_size': roi_size[0], 'frames': roi_size[-1],
        'image_patch_size': 16, 'frame_patch_size': 2,
        'dim': 1024, 'depth': 6, 'heads': 8, 'mlp_dim': 2048,
        'dropout': 0.1, 'emb_dropout': 0.1}
```

and passes `vit_settings=vit_settings` **and**
`model_config={'groups': 1, 'blocks_args': 'efficientnet-b4'}` to the framework.
These belong to the ViT and EfficientNet branches of the same API and are
**inert** when `model_name='resnet18'`: `task.json` records `vit_settings: null`
for the trained model, and the saved weights contain no Transformer tensors.
They must **not** be reported as part of the comparator.

## 4 · Provenance chain of the archived model

```
2026-10-01 03:43  split_info/train-RND-0.txt, val-RND-0.txt created (358 / 40)
2026-10-01 03:43  run_cv10_train.py written (10-fold driver, 64×64×48)   ← not used for the archive
2026-10-06 09:43  dl_models/resnet18/viz/BEST-training-params.pth written ← the archived model
2026-10-06 09:47  Step1 notebook saved; Step2 prints the loaded module summary
2026-10-06 10:50  config.yaml last edited
```

The three sibling architectures were all trained the same day
(`dl_models/resnet10`, `resnet18`, `resnet50`), each with its own `task.json`.

## 5 · The 10-fold script is a different configuration

`run_cv10_train.py` exists in the same folder but describes a **different** run
and did not produce the archived weights:

| Hyper-parameter | Notebook (archived model) | `run_cv10_train.py` |
|---|---|---|
| `roi_size` | **[48, 48, 32]** | [64, 64, 48] |
| `batch_size` | **16** | 4 |
| `init_lr` | **1 × 10⁻⁵** | 1 × 10⁻³ |
| `save_per_epoch` | **True** | False |
| `model_root` | `dl_models/` | `dl_models/fold{0..9}/` |
| Folds | 1 (`-0`) | 10 (`train-RND-0…9`) |

Two facts identify the notebook as the source of the archive: the run metadata
records 48 × 48 × 32 and 16, and there is **no** `dl_models/fold0 … fold9`
directory, so the 10-fold driver never produced output.  **Methods S4 reports
the notebook configuration**, which is the correct one to cite.

## 6 · What is *not* reproducible here

The comparator scores shipped in this repository
(`data/dl_comparator_results.csv`, `data/dl_bootstrap_ci.csv`) are the frozen
outputs.  Rebuilding them requires the **image data** (crop → resample to
1 × 1 × 1 mm → z-score over non-zero voxels), the OneKeyAI environment
(`D:\OnekeyPlatform\onekey_envs`, whose `run_classification3d.py` is
**Pyarmor-obfuscated** and cannot be read) and the trained weights — none of
which are distributable.  What *is* verifiable offline is what this page
records: the architecture, the input geometry, the optimisation settings and the
feature dimension.

## 7 · Suggested wording for Supplementary Methods S4

The current sentence

> *“3D ResNet-18 [32] (MONAI, ImageNet-initialized, `in_channels = 1`, 48 × 48 × 32
> voxels) …”*

overstates the implementation in one respect — the model is **not** MONAI's
`resnet18` (see §1).  The citation `[32]` is the correct one (He *et al.*, deep
residual learning), so only the implementation attribution needs fixing.
Suggested replacement:

> **Network and training.** A 3-D ResNet-18 with identity-shortcut basic blocks
> (3-D classification branch of OneKeyAI; `in_channels = 1`; 32,995,027
> parameters), taking 48 × 48 × 32-voxel inputs and producing 512-dimensional
> globally average-pooled features; trained separately on T2WI, DWI and ADC,
> with a fourth configuration concatenating the three feature vectors (3 × 32 =
> 96 after compression). Same 398-patient cohort; Adam (1 × 10⁻⁵); batch size
> 16; 50 epochs; a 40-patient internal validation split drawn from the training
> cohort; the best-discriminating epoch retained. Cohort small relative to
> ≈3.3 × 10⁷ parameters → single protocol, no exhaustive tuning.

Two further points to decide:

1. **“MONAI, ImageNet-initialized”** — replace `MONAI` with `OneKeyAI` (or drop
   the library name and cite `[32]` only).  The archive does *not* contain the
   initialisation routine, so “ImageNet-initialized” cannot be verified from the
   released material; if it is kept, add that the public weights are 2-D and
   were adapted to a single-channel 3-D input.
2. **Crop-and-observe** — the description “resampled to 1 × 1 × 1 mm³, z-score
   normalized over non-zero voxels” is a *pre-processing* statement that lives
   outside the notebook (`data_pattern` points at an already crop-normalised
   volume set produced by the OneKeyAI crop tool).  It is consistent with the
   in-notebook `normalize_method='imagenet'` only if the z-score step happens
   first, in the crop tool.  Worth a footnote so a reader can follow the order
   of operations.
