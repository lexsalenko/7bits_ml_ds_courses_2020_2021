import pandas as pd
import numpy as np
import cv2
import timm
import torch
import torch.nn as nn
import albumentations as A
import torchmetrics

from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold

from CustomResNet import CustomResNet
from LitCassava import LitCassava
from PlantDataset import PlantDataset

p = 'E:/Education/7bits/courses/MLandDS/ml-solutions/dl_course/homeworks/Plant Pathology 2021 - FGVC8/notebooks/models/15/'


class Model():
    def __init__(self, path=p):
        self.path = path

        res_net_model_state_dict = torch.load(path + 'plant_patology_res_net_model_15_state_dict.pt')
        res_net_model = CustomResNet(model_name='resnet50', pretrained=True)
        res_net_model.load_state_dict(res_net_model_state_dict)

        lit_model_state_dict = torch.load(path + 'pppl_model_15_state_dict.pt')
        lit_model = LitCassava(res_net_model.model)
        lit_model.load_state_dict(lit_model_state_dict)

        self.res_net_model = res_net_model
        self.lit_model = lit_model

    def get_transform(self, phase: str):
        if phase == 'train':
            return Compose([
                A.RandomResizedCrop(height=512, width=512),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            return Compose([
                A.Resize(height=512, width=512),
                A.Normalize(),
                ToTensorV2(),
            ])

    def get_prediction(self, data):

        # load data
        plant_dataset_data = PlantDataset(data, self.get_transform('valid'))
        data_loader = DataLoader(plant_dataset_data, batch_size=32, shuffle=False, num_workers=2)

        # make prediction
        # self.lit_model.cuda()
        self.lit_model.eval()

        #print('11111,111111')

        sigmoid = nn.Sigmoid()

        predictions = []
        for batch in data_loader:
            # image = batch['image'].cuda()
            image = batch['image']
            with torch.no_grad():
                outputs = self.lit_model(image)
                preds = outputs.detach().cpu()
                predictions.append(sigmoid(preds).numpy() > 0.259590)
                print(sigmoid(preds).numpy())

        #print('222222222222222')
        multi_labels = ['healthy', 'scab', 'powdery_mildew', 'frog_eye_leaf_spot', 'complex', 'rust']
        predictions = pd.DataFrame(np.concatenate(predictions).astype(np.int), columns=multi_labels)

        data.iloc[:, 2:] = predictions
        data = data.dropna(axis=1, how='all')

        # print(data)
        #
        # # print(' '.join(multi_labels[data.iloc[0, 2:] == data.iloc[0, 2:].max()]))
        #
        # print(data.loc[data.iloc[0] == 1])
        #
        # labels = []
        # for i, row in data.iloc[:, 2:].iterrows():
        #     if (row['healthy'] == 1):
        #         tmp = 'healthy'
        #     else:
        #         tmp = ' '.join(multi_labels[row == row.max()])
        #     labels.append(tmp)
        #
        # data['labels'] = labels

        #print('333333333333')
        # print(data['labels'])

        russian_names = {'scab': 'Парша яблони',
                         'powdery_mildew': 'Мучнистая роса',
                         'frog_eye_leaf_spot': 'Черная гниль или пятнистость листьев лягушки',
                         'complex': 'Комплексные заболевания у растения',
                         'rust': 'Кедрово-яблочная ржавчина',
                         'healthy': 'Полностью здоровое растение'}

        res = ''

        if data['healthy'].values == 1:
            return 'Полностью здоровое растение'
        else:
            for label in multi_labels:
                if data[label].values == 1:
                    res += russian_names[label] + ' '

        return res
