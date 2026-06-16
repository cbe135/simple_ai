# 2. 安裝與載入套件

## 安裝

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

## 載入套件

```python
import logging
import math
import os
import shutil
import yaml
import zipfile
from datetime import datetime
from tqdm.notebook import tqdm

import monai
import numpy as np
import timm
import torch
from matplotlib import pyplot as plt
from monai.bundle import ConfigParser
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose, EnsureTyped, LoadImaged,
    MaskIntensityd, RandAffined, RandFlipd,
    RepeatChanneld, Resized, ScaleIntensityd,
    ScaleIntensityRanged, ToDeviced, Transform,
    RandGaussianNoiseD,
)
from monai.utils.misc import set_determinism
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torchvision import transforms
```

## 套件版本確認

```python
print(f"MONAI version: {monai.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"timm version: {timm.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```
