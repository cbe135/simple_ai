# 環境設定指南

本教程支援三種執行環境：**Google Colab**、**Kaggle**、以及**本機（Local）**。

---

## Google Colab（推薦）

### 1. 上傳與設定

1. 點選左上角的「檔案」→「在雲端硬碟中儲存複本」
2. 點選下列連結之一，將資料新增到你的雲端硬碟：
   - [Link_0](https://drive.google.com/file/d/1LNkFfchl4YwKzLJ5SVDovhyvmw6vUUMf/view?usp=sharing)
   - [Link_1](https://drive.google.com/file/d/1vki3HykS0akuKoyLQ11yTtmucr-T4leZ/view?usp=sharing)
   - [Link_2](https://drive.google.com/file/d/1ueP6RT9NAxMO2khrqFDvIGHyCYglH0eE/view?usp=sharing)
3. 在右上角「新增雲端硬碟捷徑」到「我的雲端硬碟」

### 2. 設定 GPU

在右上角「執行階段」→「變更執行階段類型」→ 硬體加速器選「GPU」

### 3. 掛載 Google Drive

```python
from google.colab import drive
drive.mount('drive', force_remount=True)
```

### 4. 克隆 Repo 並安裝

```python
!git clone https://github.com/<your-org>/simple-ai.git
%cd simple-ai
!uv sync
```

### 5. 執行

```python
!uv run python src/main.py
```

---

## Kaggle

### 1. 上傳資料

將 `liver_data.zip` 上傳為 Kaggle Dataset，並在 Notebook 的 Input 中掛載。

### 2. 克隆 Repo

```python
!git clone https://github.com/<your-org>/simple-ai.git
%cd simple-ai
!uv sync
```

### 3. 修改資料路徑

如果你的資料在 Kaggle Input 目錄，修改 `data_dir` 指向正確路徑。

---

## 本機（Local）

### 1. 安裝 uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 克隆並安裝

```bash
git clone https://github.com/<your-org>/simple-ai.git
cd simple-ai
uv sync
```

### 3. 準備資料

資料會透過 `gdown` 自動從 Google Drive 下載。如果下載失敗，請手動下載後放在專案目錄下：

```bash
# 安裝 gdown
uv pip install gdown

# 下載資料（使用 Google Drive ID）
gdown "https://drive.google.com/uc?id=1LNkFfchl4YwKzLJ5SVDovhyvmw6vUUMf" -O liver_data.zip
```

或直接使用 Medical Segmentation Decathlon 的公開資料集。

### 4. 執行

```bash
uv run python src/main.py
```
