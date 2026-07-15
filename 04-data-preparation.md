# 4. 資料準備

## 4-1. 資料目錄格式

`main.py` 需要一個**資料目錄（data_dir）**，裡面至少要有：

```
data_dir/
├── data_list.yaml     # data: 每筆資料的 image / mask / label
├── images/            # 影像檔
└── masks/             # （可選）mask 檔
```

`data_list.yaml` 的格式如下（`modality` 不再寫在檔案裡，改用 `--modality` 傳入）：

```yaml
data:
  - image: images/001.nii.gz
    label: 1
  - image: images/002.nii.gz
    mask:  masks/002.nii.gz
    label: 0
  - image: images/003.jpg
    label: 1
```

- `--modality` 決定前處理方式（`ct` → 窗寬窗位 + mask + resize；`mri` / `xray` → 只 resize；`color` → 只 resize 且不做通道重複）。

> 沒有 `ct2d` / `ct3d` / `mri2d` / `mri3d` 這類寫法：影像是 2D 還是 3D 體積，由副檔名自動判斷（`.nii.gz` / `.nii` → 體積；`.jpg` / `.png` → 2D）。`--modality` 只描述影像類型。
- 每筆資料至少需要 `image` 與 `label`。
- `mask` 為可選；當資料中有任一筆帶 `mask` 時，pipeline 會自動啟用 mask 處理。
- 讀取器由第一個影像的副檔名自動決定（`.nii.gz` → NIfTI，`.jpg` / `.png` → PIL）。

> `data_list.yaml` 是由 `src/prepare_data.py` 自動產生的，你通常不需要手寫。

## 4-2. 下載與解壓（只需一次）

使用 `simple_ai_train_data`（對應 `src/prepare_data.py`）下載並解壓資料：

```bash
uv run simple_ai_train_data --data-dir /content/liver_data --file-ids 1LNkF...

# 也可直接用 Google Drive 資料夾 ID
uv run simple_ai_train_data --data-dir /content/liver_data --gdown-id ABCD1234

# 指定壓縮格式（預設 zip）
uv run simple_ai_train_data --data-dir /content/liver_data --file-ids 1LNkF... --archive-format zip
```

參數：

- `--data-dir`（必填）：輸出資料目錄。
- `--file-ids`：Google Drive 檔案 ID（可用逗號串多個）。
- `--gdown-id`：Google Drive 資料夾 ID。
- `--archive-format`：壓縮格式（`zip` / `tar` / `tar.gz`）。
- `--data-name`：內部暫存名稱（預設 `liver_data`）。
- `--config`：對應的設定檔（預設 `config.yaml`）。

下載後會自動解壓、掃描 `images/`、並寫出 `data_list.yaml`
（`modality` 不寫入檔案，訓練時用 `--modality` 傳入）。

> **冪等**：若 `data_dir/data_list.yaml` 已存在，`simple_ai_train_data` 會直接
> 跳過（不重複下載），方便在 notebook 重跑 cell 時不浪費時間。

## 4-3. 在程式中使用

```python
from src.data import load_data_list, populate_data_lists

data_dicts = load_data_list(args, "/content/liver_data")
# data_dicts 為上面 data 清單；modality 由 --modality 傳入
args.data_list = populate_data_lists(args, data_dicts)
```

## 4-4. 執行訓練時帶入資料目錄

`--data-dir` 與 `--modality` 都是 `simple_ai_train` 的**必填**參數：

```bash
uv run simple_ai_train --data-dir /content/liver_data --modality ct
```
