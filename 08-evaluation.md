# 8. 檢視成果

訓練完模型之後，我們來看一下訓練的成果。
把模型輸出值變成 0 跟 1，跟標準答案比較，產出混淆矩陣的值，靈敏度，特異度。

## 8-1. 載入最佳權重並推論

```python
import torch
from src.evaluate import infer

best_state = torch.load("/content/best_weights.pth", weights_only=True)
model.load_state_dict(best_state)

train_true, train_pred = infer(args, model, train_loader, True)
val_true, val_pred = infer(args, model, val_loader, True)
test_true, test_pred = infer(args, model, test_loader, True)
```

## 8-2. Grad-CAM 視覺化

Grad-CAM 可以幫助我們了解模型關注影像的哪些區域。

```python
from src.evaluate import grad_cam

grad_cam(
    model,
    "/content/liver_data/images/liver_118_13.nii.gz",
    class_id=0,
    args=args,
)
```

## 8-3. ROC 曲線與評估指標

```python
from src.evaluate import plot_roc_and_show_result

plot_roc_and_show_result(args, train_true, train_pred, title='Train')
plot_roc_and_show_result(args, val_true, val_pred, title='Validation')
plot_roc_and_show_result(args, test_true, test_pred, title='Test')
```

## 評估指標說明

| 指標 | 公式 | 說明 |
|------|------|------|
| Sensitivity（靈敏度） | TP / (TP + FN) | 正確預測正樣本的比例 |
| Specificity（特異度） | TN / (TN + FP) | 正確預測負樣本的比例 |
| AUC | ROC 曲線下面積 | 模型整體分類能力 |
