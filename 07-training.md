# 7. 訓練

準備好資料之後，我們來建立一個 CNN 模型來判斷影像中的病人是否有病灶（二元分類），
或做多類別分類。

## 7-1. 模型與優化器

`src/model.py`：

| 函式 | 作用 |
|---|---|
| `create_timm_model(args)` | 依 `args.timm_model` 建立模型（預設 `resnet18`），輸出維度由 `args.num_classes` 決定 |
| `generate_optimizer(args, model)` | 依 `args.optimizer.name`（`adam` / `sgd`）建立優化器，學習率 `args.lr` |
| `get_device()` | 選擇 `cuda` 或 `cpu` |

```python
from src.model import create_timm_model, generate_optimizer, get_device

device = get_device()
model = create_timm_model(args).to(device)
optimizer = generate_optimizer(args, model)
```

## 7-2. 損失函數與訓練迴圈

`src/train.py`：

| 函式 | 作用 |
|---|---|
| `build_criterion(args)` | 依 `args.loss.name` 建立損失（`bce_with_logits` → `BCEWithLogitsLoss`；`ce` → `CrossEntropyLoss`） |
| `train_one_epoch(...)` | 單一 epoch 的前向 / 反向傳播 |
| `train(args, model, train_loader, val_loader, criterion, optimizer, device, num_epoch, run_dir)` | 完整訓練 + 每 epoch 在驗證集評估 |
| `train_pipeline(args, train_set, val_set, run_dir, device)` | 把「模型 + 優化器 + 損失 + DataLoader + 訓練」串起來，並把權重 / loss 曲線存到 `run_dir` |

```python
from src.train import build_criterion, train_pipeline

criterion = build_criterion(args)
run_dir = train_pipeline(args, train_set, val_set, run_dir, device)
```

## 7-3. 執行訓練（指令列）

`batch_size` 預設為 **16**（見 `config.yaml` 的 `training.batch_size`）：

```bash
uv run simple_ai_train --data-dir /content/liver_data

# 指定設定檔與輸出目錄
uv run simple_ai_train --config config.yaml --data-dir /content/liver_data --output-dir ./results
```

訓練產物會寫入時間戳記的 `run_dir`：

```
runs/<timestamp>/
├── best_model.pth          # 驗證集最佳權重
├── loss_curve.png          # 訓練 / 驗證 loss 曲線
├── config.yaml             # 本次執行使用的設定
├── samples/                # 前處理前後對照（見 05）
├── inference/              # 推論細節（見 08）
└── roc/                    # ROC 曲線（見 08）
```

## 7-4. 在程式中使用

```python
from src.main import get_parser, main
# 直接呼叫 main（會依 parser 讀取 --config / --data-dir）
```

或直接用各模組組裝你自己的迴圈：

```python
from src.data import load_modality_and_data, populate_data_lists, generate_dataset, generate_dataloader
from src.transforms import build_train_transform, build_val_transform
from src.model import create_timm_model, generate_optimizer, get_device
from src.train import build_criterion, train_pipeline

modality, data_dicts = load_modality_and_data(args.data_dir)
args.data_list = populate_data_lists(args, data_dicts)
from src.transforms import derive_spatial_dims
spatial_dims = derive_spatial_dims(data_dicts[:1])
dataset_info = {"modality": modality, "spatial_dims": spatial_dims}

train_tf = build_train_transform(args, data_dicts[:1], dataset_info)
val_tf   = build_val_transform(args, data_dicts[:1], dataset_info)
train_set = generate_dataset(args, args.data_list["train"], train_tf)
val_set   = generate_dataset(args, args.data_list["val"],   val_tf)

device = get_device()
model = create_timm_model(args).to(device)
optimizer = generate_optimizer(args, model)
criterion = build_criterion(args)
run_dir = train_pipeline(args, train_set, val_set, run_dir, device)
```
