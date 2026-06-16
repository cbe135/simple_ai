# D3 Hands-on：CT 影像肝臟腫瘤分類（CNN）

**2026 Winter — 最後修改日期：2025/12/17**

## CNN Classification

**Dataset reference:** [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

本教程將帶領你使用 CNN（Convolutional Neural Network）對 CT 影像中的肝臟進行分類，判斷影像中的病人是否有肝腫瘤。

### 學習目標

- 了解醫學影像分類的基本流程
- 學會使用 MONAI 進行資料前處理與增強
- 使用 PyTorch + timm 建立 CNN 分類模型
- 評估模型表現（ROC 曲線、混淆矩陣、Grad-CAM）

### 使用的工具與框架

| 工具 | 用途 |
|------|------|
| MONAI | 醫學影像前處理、資料載入 |
| PyTorch | 深度學習框架 |
| timm | 預訓練模型（ResNet-18） |
| scikit-learn | 評估指標 |

### 環境需求

- Python >= 3.10
- GPU 建議使用（Colab 的 T4 或更好的 GPU）

> 下一節：[環境設定](environment.md)
