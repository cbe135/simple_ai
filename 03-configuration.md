# 3. 設定參數

我們使用一個 YAML 設定檔（預設 `config.yaml`）來管理所有參數，傳入各模組的
`args`。參數分為幾個主要區塊：`environ`、`data`、`training`、`threshold`、
`modalities`、`transforms`。

> **重要**：`modality`（`ct` / `mri` / `xray` / `color`）**是** `config.yaml` 裡
> `modalities` 區塊的**鍵**，每一個鍵定義該影像類型的完整前處理與增強
> （MONAI bundle `_target_` 清單）。訓練時用 `--modality` 參數指定要使用哪一個
> 鍵；`--modality` 的合法值就是 `modalities` 的所有鍵。
> 這就是「資料驅動」的核心：在 `config.yaml` 調整 `modalities` 或 `data`，
> 行為就跟著變，不需要改任何程式碼。

## config.yaml 結構

```yaml
environ:
  config_file: config.yaml
  seed: 888

data:
  train_percentage: 0.8
  val_percentage: 0.1
  test_percentage: 0.1
  spatial_size: [250, 250]      # 所有影像 resize 到此尺寸（2D）；若資料被自動偵測為 3D，會自動補成 [250, 250, 250]
  repeats: 3                    # 單通道影像重複成多通道
  rotate_range: [[0.17, 0.35], [0.17, 0.35]]
  shear_range: [[0, 0], [0, 0]]
  translate_range: [[-60, 60], [0, 0]]
  scale_range: [[0, 0], [0, 0]]
  affine_prob: 0                # RandAffine 機率（0 = 關閉）
  spatial_axis: [0, 1]
  flip_prob: 0.5                # RandFlip 機率
  a_min: -125                   # CT 窗位下限（HU）
  a_max: 200                    # CT 窗位上限（HU）
  cache_rate: 1.0
  num_workers: 4

training:
  num_epoch: 3
  batch_size: 16
  lr: 0.001
  timm_model: resnet18
  num_classes: 1
  optimizer:
    name: adam
    weight_decay: 0.0
    momentum: 0.9
  loss:
    name: bce_with_logits

threshold: 0.5

transforms:                     # 額外 transform（MONAI bundle 格式）
  loaders_extra: []
  preprocess_extra: []
  augmentation_extra: []
```

### 各區塊說明

- **environ**：`seed` 用於重現性；`config_file` 記錄目前使用的設定檔。
- **data**：資料與前處理相關參數。`spatial_size` 決定輸入尺寸；
  `a_min` / `a_max` 是 CT 窗位；`rotate_range` / `shear_range` /
  `translate_range` / `scale_range` / `affine_prob` 控制 `RandAffine`；
  `flip_prob` 控制 `RandFlip`；`num_workers` / `cache_rate` 是 DataLoader 設定。
- **training**：`batch_size` 預設 **16**（不是早期筆記本的 128）；
  `timm_model` 可用任意 timm 模型名稱；`num_classes` 決定輸出維度；
  `optimizer.name` 支援 `adam` / `sgd`；`loss.name` 支援 `bce_with_logits` /
  `ce`（見 `src/train.py` 的 `build_criterion`）。
- **threshold**：推論時將機率轉成預測類別的門檻。
- **transforms**：在自動推導出的 preset transform 之後，額外附加的 MONAI transform
  （`loaders_extra` / `preprocess_extra` / `augmentation_extra`）。

## 執行方式

```bash
# 預設讀取 config.yaml
uv run simple_ai_train --data-dir /content/liver_data

# 指定設定檔
uv run simple_ai_train --config config.yaml --data-dir /content/liver_data
```

> `main.py` 會在 `run_dir/config.yaml` 存一份本次執行使用的設定（`run_dir`
> 為時間戳記目錄，例如 `runs/20260106_153000/`）。CLI 模式下並不會另外寫出
> 時間戳記命名的 `config.yaml` 到側邊欄；請直接用 `--config` 指定檔名。

## 在程式中使用

`src/config.py` 的 `get_config(parser)` 會把 YAML 讀進 `argparse.Namespace`
（`args`），各模組都直接接收這個 `args`：

```python
from src.config import get_config
from src.main import get_parser

parser = get_parser()
args, _ = parser.parse_known_args()
# args.data_dir, args.batch_size, args.timm_model, ...
```
