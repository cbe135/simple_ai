# 8. 檢視成果

訓練完模型之後，我們來看一下訓練的成果：推論結果、ROC 曲線、混淆矩陣與 Grad-CAM。

## 8-1. 推論（infer）

`src/evaluate.py` 的 `infer(args, model, data_loader, details, device, details_path)`
會在測試集上跑推論，並把每一筆的機率與預測寫入 `run_dir/inference/`
（原 `inference_details_*.log`）：

```python
from src.evaluate import infer
from src.utils import load_state_dict

model = load_state_dict(model, f"{run_dir}/best_model.pth")
details = infer(args, model, test_loader, [], device, f"{run_dir}/inference/inference_details_test.log")
```

`details` 是 list of `(y_true, y_pred, y_prob, img_path, pred_label)`，
供後續畫 ROC / 混淆矩陣使用。

## 8-2. ROC 與混淆矩陣（plot_roc_and_show_result）

`plot_roc_and_show_result(args, y_true, y_pred, title, save_path)` 會：

- 計算 AUC 與 ROC 曲線，並把圖存到 `run_dir/roc/`（原 `roc_*.png`）。
- 計算混淆矩陣（TP / FP / FN / TN）與 sensitivity / specificity，
  透過 `logger.info` 記錄到 `run_dir/pipeline.log` 與 `run.log`
  （`src/evaluate.py` 的 `evaluate` 函式）。

```python
from src.evaluate import plot_roc_and_show_result

y_true = [d[0] for d in details]
y_pred = [d[1] for d in details]
plot_roc_and_show_result(args, y_true, y_pred, "Test ROC", f"{run_dir}/roc/roc_test.png")
```

> 混淆矩陣的數字（TP/FP/FN/TN + sensitivity/specificity）是寫進 log 的，
> 不在 standalone 的圖檔裡；要在 `run_dir/pipeline.log` 或 `run.log` 查看。

## 8-3. Grad-CAM

`grad_cam(model, img_path, class_id, args, data_list, device, save_path)`
產生類別激活圖，存到指定路徑：

```python
from src.evaluate import grad_cam

grad_cam(model, "images/img_001.nii.gz", 1, args, args.data_list["test"], device,
         f"{run_dir}/gradcam_img_001.png")
```

## 8-4. 執行方式（指令列）

評估是訓練流程的一部分，`simple_ai_train` 會自動在測試集上推論並產生
`inference/` 與 `roc/` 輸出，不需要額外指令：

```bash
uv run simple_ai_train --data-dir /content/liver_data
```

`run_dir` 的最終結構：

```
runs/<timestamp>/
├── best_model.pth
├── loss_curve.png
├── config.yaml
├── pipeline.log
├── samples/        # 前處理前後對照
├── inference/      # inference_details_*.log
└── roc/            # roc_*.png
```
