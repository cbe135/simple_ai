# 3. 設定參數

我們使用一個字典 `args` 來管理所有參數。參數分為幾個主要區塊：

## 參數設定

```python
import numpy as np
from datetime import datetime
import yaml

args = {
    "environ": {
        "config_file": "config.yaml",
        "seed": 888,
        "data_name": "liver_data",
    },

    "data": {
        "train_percentage": 0.8,
        "val_percentage": 0.1,
        "test_percentage": 0.1,

        "spatial_size": [250, 250],
        "repeats": 3,

        "rotate_range": [[np.pi/18, np.pi/9], [np.pi/18, np.pi/9]],
        "shear_range": [[0, 0], [0, 0]],
        "translate_range": [[-60, 60], [0, 0]],
        "scale_range": [[0, 0], [0, 0]],
        "affine_prob": 0,

        "spatial_axis": [0, 1],
        "flip_prob": 0.5,

        'a_min': -125,
        'a_max': 200,

        "cache_rate": 1,
    },

    "img_cnt": 5,

    "training": {
        "num_epoch": 3,
        "batch_size": 128,
        "lr": 1e-3,
        'timm_model': 'resnet18',
        'num_classes': 1,
    },

    "threshold": 0.5,

    # 額外的 transform（MONAI bundle 格式），附加在自動推導的預設 transform 之後
    "transforms": {
        "loaders_extra": [],
        "preprocess_extra": [],
        "augmentation_extra": [],
    },
}

# 儲存設定
with open(f'{datetime.strftime(datetime.now(), "%m%d_%H%M%S")}_{args["environ"]["config_file"]}', 'w') as fp:
    yaml.dump(args, fp)
```

## 參數說明

| 參數 | 說明 |
|------|------|
| `seed` | 隨機種子，確保可重複性 |
| `data_name` | 資料夾名稱 |
| `train/val/test_percentage` | 資料切分比例 |
| `spatial_size` | 影像調整後大小 |
| `a_min/a_max` | CT 開窗範圍（HU 值） |
| `num_epoch` | 訓練輪數 |
| `batch_size` | 批次大小 |
| `lr` | 學習率 |
| `timm_model` | 使用的預訓練模型 |
| `threshold` | 二分類閾值 |

## Transforms（額外 transform）

預設的 transform（讀取、前處理、增強）會根據資料自動推導（`dataset_info.yaml` 的 `modality`、是否有 mask、檔案副檔名）。你可以在 `transforms` 區塊用 MONAI bundle 格式追加自己的 transform：

```yaml
transforms:
  loaders_extra: []        # 附加在 LoadImaged / EnsureTyped 之後
  preprocess_extra: []     # 附加在 Resize / 開窗 / MaskIntensity / RepeatChannel 之後
  augmentation_extra: []   # 附加在 RandAffine / RandFlip / GaussianNoise 之後（僅訓練）
```

可使用 `@data.*` 參考上方 `data` 區塊的參數，例如：

```yaml
transforms:
  preprocess_extra:
    - _target_: monai.transforms.RandGaussianSmoothd
      keys: ["image"]
      sigma_x: [0.5, 1.0]
```

`_target_` 請使用完整路徑（如 `monai.transforms.RandFlipd`）。留空則只使用預設 transform。

## 環境設定

```python
import logging
from monai.utils.misc import set_determinism

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger()

set_determinism(args["environ"]["seed"])
```
