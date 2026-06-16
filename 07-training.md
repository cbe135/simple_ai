# 7. 訓練

準備好資料之後，我們來建立一個 CNN 模型來判斷 CT 影像中的病人是否有肝癌。
這裡我們使用 PyTorch 的 ResNet-18，也可以嘗試使用不同的 [模型](https://pytorch.org/vision/stable/models.html#classification)。
大部分模型的權重都是用 [ImageNet](https://www.image-net.org/) 訓練出的結果。

> 補充：ResNet-18 接受的是 (3, 224, 224) 大小的影像，因此在前處理時把調整大小的參數設定成 `spatial_size=[250, 250]`。

## 7-1. 建立模型

```python
from src.model import create_timm_model, get_device
from src.data import generate_dataloader

device = get_device()
model = create_timm_model(args).to(device)

train_loader = generate_dataloader(args, train_set, shuffle=True)
val_loader = generate_dataloader(args, val_set)
test_loader = generate_dataloader(args, test_set)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args["training"]["lr"])
```

## 7-2. 訓練模型

在這裡開始訓練模型，這可能會跑一段時間。

```python
%%time
from src.train import train

record = train(args, model, criterion, optimizer, train_loader, val_loader)
```

## 7-3. 觀察 Loss 曲線

在訓練完模型之後，我們可以觀察 loss 值在訓練集和驗證集上的表現。

```python
from src.utils import plot_loss_curves

plot_loss_curves(args, record)
```
