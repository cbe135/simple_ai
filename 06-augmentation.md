# 6. 資料增強

## 6-1. 增強方式說明

我們可以執行資料增強來增加資料的多樣性：

- **旋轉**
- **裁減**
- **平移**
- **放大縮小**
- **翻轉**

更多 MONAI 增強方式請參考：[MONAI Transforms](https://docs.monai.io/en/stable/transforms.html)

## 6-2. 隨機仿射變換

```python
from src.utils import plot_transform_result
from monai.transforms import RandAffined

affine = RandAffined(
    keys='image',
    rotate_range=args["data"]['rotate_range'],
    shear_range=args["data"]['shear_range'],
    translate_range=args["data"]['translate_range'],
    scale_range=args["data"]['scale_range'],
    prob=args["data"]['affine_prob'],
    padding_mode='border'
)

affine_data = affine(data)
plot_transform_result(data, affine_data)
```

## 6-3. 隨機翻轉

```python
from monai.transforms import RandFlipd

flipper = RandFlipd(keys='image', spatial_axis=args["data"]["spatial_axis"], prob=args["data"]["flip_prob"])
flipped_data = flipper(data)

plot_transform_result(data, flipped_data)
```

## 6-4. 載入並設定資料

```python
from src.data import generate_dataset, generate_dataloader
from src.transforms import build_train_transform, build_val_transform

train_transform = build_train_transform(args)
val_transform = build_val_transform(args)

train_set = generate_dataset(args, train_dicts, train_transform)
val_set = generate_dataset(args, val_dicts, val_transform)
test_set = generate_dataset(args, test_dicts, val_transform)
```

## 6-5. 檢視資料分佈

```python
from src.data import check_dist

check_dist(train_set)
check_dist(val_set)
check_dist(test_set)
```

## 6-6. 檢視訓練集影像

```python
sample_data = [train_set[i] for i in random_idx]
plot_samples(sample_data)
```
