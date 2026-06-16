# 5. 前處理

## 5-1. 前處理步驟說明

我們需要載入資料並進行前處理：

1. 把資料整理成模型需要的格式
2. 去除沒有用的雜訊

以下是會使用到的前處理步驟：

- **調整大小**：調整圖像到特定大小。模型接受的影像大小是 (250, 250)。
- **開窗**：調整影像的亮度和對比。這前處理只適合 CT。可調整窗寬和窗位。
- **資料正規化**：將每一個像素的值調整到 0 跟 1 之間。
- **去背**：將肝臟外的背景設為 0。

## 5-2. 示範：調整大小

```python
from src.utils import plot_transform_result
from monai.transforms import Resized

data = load(data_dicts[30])
print(data['image'].size(), data['mask'].size())

resizer = Resized(keys=['image', 'mask'], spatial_size=args["data"]["spatial_size"])
resized_data = resizer(data)

plot_transform_result(data, resized_data, with_mask=True)
```

## 5-3. 示範：正規化

```python
from monai.transforms import ScaleIntensityd

normalizer = ScaleIntensityd(keys='image', minv=0.0, maxv=1.0)
normalized_data = normalizer(data)

plot_transform_result(data, normalized_data, with_histogram=True)
```

## 5-4. 示範：開窗

```python
from monai.transforms import ScaleIntensityRanged

windower = ScaleIntensityRanged(
    keys=['image'],
    a_min=args["data"]['a_min'],
    a_max=args["data"]['a_max'],
    b_min=0, b_max=1,
    clip=True
)
windowed_data = windower(data)
plot_transform_result(data, windowed_data, with_histogram=True)
```

## 5-5. 示範：去背

```python
from monai.transforms import MaskIntensityd

masker = MaskIntensityd(keys='image', mask_key='mask')
masked_data = masker(data)

plot_transform_result(data, masked_data, with_mask=True)
plot_transform_result(data, masked_data, with_histogram=True)
```

## 5-6. 完整前處理管線

```python
from src.transforms import get_loaders, get_preprocess, get_augmentation
from monai.transforms import Compose

loaders = get_loaders()
preprocess = get_preprocess(args)
augmentation = get_augmentation(args)

train_transform = Compose(loaders + preprocess + augmentation)
val_transform = Compose(loaders + preprocess)
```
