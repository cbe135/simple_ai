# 6. 資料增強

## 6-1. 增強方式說明

增強由 `config.yaml` 的 `modalities.<modality>.augmentation` 定義
（MONAI bundle `_target_` 清單），並由 `src/transforms.py` 的
`get_modality_pipeline` 讀出；只會出現在**訓練** transform
（驗證 / 測試不增強）。設計上刻意**不使用 crop**，以保留影像完整資訊。

預設每一個 modality 的 augmentation 清單包含：

- `RandAffined`：依 `args` 中的 `rotate_range` / `shear_range` /
  `translate_range` / `scale_range` / `spatial_axis` 做隨機仿射，
  觸發機率由 `affine_prob` 控制（預設 `0`，即關閉）。
- `RandFlipd`：`flip_prob`（預設 `0.5`）做隨機水平翻轉。
- `RandGaussianNoised`：**永遠**啟用，對影像加微小高斯雜訊，提升穩健性。

這些都寫在 `config.yaml` 的 `modalities.<modality>.augmentation` 裡，
調整時直接改 config（見 `03-configuration.md` 與 `src/program.md`）。

```python
from src.transforms import (
    get_modality_pipeline, build_train_transform, build_val_transform,
)

dataset_info = {"modality": modality, "spatial_dims": spatial_dims}
preprocess, augmentation = get_modality_pipeline(args, data_dicts[:1], dataset_info)

# 訓練 transform = loaders + loaders_extra + preprocess + preprocess_extra
#                  + augmentation + augmentation_extra + strip_meta
train_tf = build_train_transform(args, data_dicts[:1], dataset_info)
# 驗證 transform = loaders + loaders_extra + preprocess + preprocess_extra
#                  + strip_meta（沒有 augmentation）
val_tf   = build_val_transform(args, data_dicts[:1], dataset_info)
```

## 6-2. 資料集與 DataLoader

`src/data.py` 負責把 transform 套用到資料並建立 DataLoader：

| 函式 | 作用 |
|---|---|
| `generate_dataset(args, datalist, transform)` | 建立 `Dataset` |
| `generate_dataloader(args, dataset, shuffle, device)` | 建立 `DataLoader`（`batch_size` 來自 `args.batch_size`，預設 16） |
| `check_dist(dataset)` | 檢查每類別樣本數是否為正數（避免空類別） |

```python
from src.data import generate_dataset, generate_dataloader, check_dist

train_set = generate_dataset(args, args.data_list["train"], train_tf)
check_dist(train_set)                       # 確認類別分布合理
train_loader = generate_dataloader(args, train_set, shuffle=True, device=device)
```

## 6-3. 自訂增強（transforms 區塊）

若 preset 不夠，可在 `config.yaml` 的 `transforms.augmentation_extra` 用
MONAI bundle 格式附加 transform：

```yaml
transforms:
  augmentation_extra:
    - _target_: monai.transforms.RandGaussianSmoothd
      keys: ["image"]
      sigma_x: [0.5, 1.0]
```

它會被附加在該 modality 的 preset 增強之後。更多欄位請見 `03-configuration.md`。
