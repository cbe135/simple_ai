# 5. 前處理

## 5-1. 前處理步驟說明

前處理由 `src/transforms.py` 自動推導，不需要手寫 `if modality == ...`。
核心函式如下：

| 函式 | 作用 |
|---|---|
| `derive_reader(data_dicts_sample)` | 由第一個影像副檔名決定讀取器（`.nii.gz` → NIfTI，`jpg/png` → PIL） |
| `derive_has_masks(data_dicts_sample)` | 由資料中是否含 `mask` 鍵決定是否啟用 mask |
| `get_loaders(data_dicts_sample)` | 產生 `LoadImaged` / `EnsureTyped` + 可選 `LoadImaged(mask)` |
| `get_preprocess(args, data_dicts_sample, dataset_info)` | 依 `modality` 產生前處理（`ct` → `ScaleIntensityRanged` 窗位 + 可選 `MaskIntensityd` + `Resized` + `RepeatChanneld`；`mri` / `xray` → `Resized` + `RepeatChanneld`；`color` → `Resized` 不含 `RepeatChanneld`） |
| `get_augmentation(args, dataset_info)` | 產生增強 transform（見 06） |
| `build_train_transform(args, data_dicts_sample, dataset_info)` | `loaders + preprocess + augmentation + 額外` 組成訓練 transform |
| `build_val_transform(args, data_dicts_sample, dataset_info)` | `loaders + preprocess + 額外`（**不含**增強）組成驗證 transform |

其中 `dataset_info` 是一個 dict：`{"modality": "ct", "spatial_dims": 2}`（大小寫不敏感），
由 `data.py` 的 `load_modality_and_data` 讀取 `data_list.yaml` 頂層 `modality` 得到；
`spatial_dims` 則由 `transforms.derive_spatial_dims` **載入第一張影像後依其形狀自動判斷**
（`.nii.gz` 既可能是 2D 也可能是 3D，無法只看副檔名）。

> `spatial_size` 與各 affine 範圍（`rotate_range` / `shear_range` /
> `translate_range` / `scale_range`）都會**自動補齊 / 截斷**到偵測到的維度：
> 2D 設定遇到 3D 資料時，`spatial_size` 補成三等邊（如 `[250,250,250]`），
> affine 範圍在新軸上補 `[0, 0]`（該軸不做增強）。

## 5-2. CT 的窗寬窗位

CT 的 HU 值範圍很大，先用 `ScaleIntensityRanged`
（參數 `a_min` / `a_max`，預設 `-125` ~ `200`）把感興趣的範圍線性映射到
`[0, 1]`，超過範圍的值會被 clip，這能強化肝臟腫瘤的對比。

```python
from src.transforms import (
    derive_reader, derive_has_masks, get_loaders,
    get_preprocess, build_train_transform, build_val_transform,
)

dataset_info = {"modality": modality}
loaders = get_loaders(data_dicts[:1])
preprocess = get_preprocess(args, data_dicts[:1], dataset_info)
train_tf = build_train_transform(args, data_dicts[:1], dataset_info)
val_tf   = build_val_transform(args, data_dicts[:1], dataset_info)
```

## 5-3. 視覺化前 / 後處理結果

`src/utils.py` 的 `plot_transform_result` 與 `plot_samples` 接受 `save_path`，
`src/main.py` 會把每個訓練 / 驗證樣本的前處理前後對照圖，存到
`run_dir/samples/` 資料夾：

```
runs/<timestamp>/
└── samples/
    ├── train_sample_0.png
    ├── val_sample_0.png
    └── ...
```

```bash
# 訓練時會自動產生，無需手動呼叫
uv run simple_ai_train --data-dir /content/liver_data
```

## 5-4. 在程式中使用

```python
from src.data import generate_dataset

train_set = generate_dataset(args, args.data_list["train"], train_tf)
val_set   = generate_dataset(args, args.data_list["val"],   val_tf)
```
