import pandas as pd
import numpy as np
import cv2
import timm
import torch
import torch.nn as nn


class CustomResNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
#         self.model.fc = nn.Linear(in_features, CFG.num_classes)
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, 6)
        )

    def forward(self, x):
        x = self.model(x)
        return x