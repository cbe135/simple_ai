# 4. 資料準備

## 4-1. 解壓縮資料

資料從 Google Drive（Colab）或直接下載（Local/Kaggle）後解壓縮。

```python
from src.env_setup import detect_environment, setup_data

logger.info(f"Environment: {detect_environment()}")
setup_data(args)
```

## 4-2. 檢視資料量

```python
from src.env_setup import get_data_count

get_data_count(args)
```

## 4-3. 載入資料列表

```python
from src.data import load_data_list, populate_data_lists

data_dicts = load_data_list(args)
print(f"Total data: {len(data_dicts)}")
print(f"Sample: {data_dicts[0]}")
```

## 4-4. 切分訓練、驗證、測試集

```python
train_dicts, val_dicts, test_dicts = populate_data_lists(args, data_dicts)

logger.info(f'{len(train_dicts)} data for training')
logger.info(f'{len(val_dicts)} data for validation')
logger.info(f'{len(test_dicts)} data for testing')
```

## 4-5. 檢視隨機樣本

```python
import numpy as np
from src.utils import plot_samples
from monai.transforms import Compose, LoadImaged, EnsureTyped

random_idx = np.random.randint(0, len(train_dicts), args["img_cnt"])
sample_data = [train_dicts[i] for i in random_idx]

load = Compose([
    LoadImaged(keys=['image', 'mask'], ensure_channel_first=True),
    EnsureTyped(keys=["image", "label"])
])

sample_data = load(sample_data)
plot_samples(sample_data, with_mask=True)
```
