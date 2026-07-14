# CNN 影像分類：CT 肝臟腫瘤分類（Hands-on）

**2026 Winter — 最後修改日期：2025/12/17**

## 這個專案在做什麼

本專案是一個**資料驅動**的醫學影像分類流程，使用 MONAI + PyTorch + timm 實作
CNN 影像分類（二元 / 多類別皆可）。整個流程的所有行為都來自設定檔與資料本身，
程式碼中**沒有任何寫死的任務名稱**（沒有 `if task == ...`）。

學習目標：

- 了解一條完整的深度學習訓練管線（資料 → 前處理 → 增強 → 訓練 → 評估）。
- 學會用設定檔（`config.yaml`）控制超參數，而不用改程式碼。
- 理解「資料驅動」的設計：前處理、讀取器、是否有 mask，全部由資料自動推導。

## 工具鏈

- **MONAI**：醫學影像的讀取與前處理。
- **PyTorch / torchvision**：深度學習框架。
- **timm**：現成模型（如 ResNet-18）。
- **scikit-learn**：評估指標（ROC、AUC、混淆矩陣）。
- **matplotlib**：視覺化。
- **gdown**：從 Google Drive 下載資料。
- **openai / python-dotenv**：autoresearch 用的 LLM 客戶端。

## 如何執行（指令列）

本專案以模組化的指令列腳本執行，不需要手動在 notebook 裡一個 cell 一個 cell 呼叫：

```bash
# 安裝相依（只需一次）
uv sync

# 訓練（資料目錄必填）
uv run simple_ai_train --data-dir /content/liver_data

# 資料準備（從 Google Drive 下載並解壓，只需一次）
uv run simple_ai_train_data --data-dir /content/liver_data --file-ids 1LNkF...

# 自動搜尋更好的 config（本地 Ollama 或 OpenRouter）
uv run simple_ai_autoresearch_train --data-dir /content/liver_data --runs 12
```

對應的程式碼模組都放在 `src/` 下（見 `README.md` 的 Project Structure），
後續章節會依模組逐一說明。
