import pandas as pd
import numpy as np
import cv2
import timm
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchmetrics
import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold

photo_path = 'E:/Education/7bits/courses/MLandDS/ml-solutions/dl_course/homeworks/Plant Pathology 2021 - FGVC8/notebooks/photo/'

class PlantDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_id = df['image'].values
        self.labels = df.iloc[:, 2:].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_id = self.image_id[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        image_path = photo_path + image_id[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image = augmented['image']
        return {'image':image, 'target': label}
        #return {'image':image}

