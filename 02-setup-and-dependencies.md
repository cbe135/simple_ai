# 2. 安裝與載入套件

## 安裝

本專案使用 [uv](https://docs.astral.sh/uv/) 管理環境（也可用 pip）。

```bash
git clone https://github.com/cbe135/simple_ai.git
cd simple_ai

# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -e .
```

`uv sync` 會根據 `pyproject.toml` 建立虛擬環境並安裝所有相依套件。

## 指令列腳本（console scripts）

安裝後，下列指令可直接在終端機 / notebook cell 中使用（`pyproject.toml` 定義）：

| 指令 | 對應模組 | 說明 |
|---|---|---|
| `simple_ai_train` | `src/cli.py` → `src/main.py` | 端到端訓練（讀取 `config.yaml` + `--data-dir`） |
| `simple_ai_train_data` | `src/prepare_data.py` | 下載並解壓資料（只需一次） |
| `simple_ai_autoresearch_train` | `src/autoresearch_cli.py` | LLM 自動搜尋更好的 config |
| `simple_ai_autoresearch_setup` | `src/autoresearch_setup.py` | 安裝 Ollama、檢查 GPU/驅動、預拉模型 |
| `simple_ai_autoresearch_serve` | `src/autoresearch_serve.py` | 背景啟動 Ollama server（可對外暴露） |

基本的訓練執行方式：

```bash
uv run simple_ai_train --data-dir /content/liver_data
# 或指定設定檔
uv run simple_ai_train --config config.yaml --data-dir /content/liver_data
```

## 套件說明

程式內部實際使用的套件（對應 `pyproject.toml` 的 dependencies）：

- `monai`：醫學影像讀取與前處理。
- `torch` / `torchvision`：模型與訓練。
- `timm`：預訓練模型（預設 `resnet18`）。
- `scikit-learn`：評估指標。
- `matplotlib`：繪圖。
- `gdown`：Google Drive 下載。
- `openai` / `python-dotenv`：autoresearch 的 LLM 客戶端。

> 注意：程式內部使用的是 `tqdm.auto`（可同時在終端機與 notebook 運作），
> 並非 `tqdm.notebook`，因此在無介面（headless）執行時也能正常顯示進度。

## 也可以直接 import 模組

如果你想在 notebook 中手動呼叫各個階段，模組都可直接 import：

```python
from src.data import load_modality_and_data, populate_data_lists, generate_dataset
from src.transforms import build_train_transform, build_val_transform
from src.model import create_timm_model, generate_optimizer
from src.train import train_pipeline, build_criterion
from src.evaluate import infer, plot_roc_and_show_result
```
