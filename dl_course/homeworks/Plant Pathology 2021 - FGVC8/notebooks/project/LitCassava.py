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

# from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning import Callback
# from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from sklearn.model_selection import StratifiedKFold


class LitCassava(pl.LightningModule):
    def __init__(self, model):
        super(LitCassava, self).__init__()
        self.model = model
        self.metric = pl.metrics.F1(num_classes=6)
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.lr = 1e-4

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    # def configure_optimizers(self):
    #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    #     self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
    #                                                          epochs=CFG.num_epochs, steps_per_epoch=CFG.steps_per_epoch,
    #                                                          max_lr=CFG.max_lr, pct_start=CFG.pct_start,
    #                                                          div_factor=CFG.div_factor, final_div_factor=CFG.final_div_factor)


    #     return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    # def training_step(self, batch, batch_idx):
    #     image = batch['image']
    #     target = batch['target']
    #     output = self.model(image)
    #     # output = self.sigmoid(output)
    #     # loss = self.criterion(output, target)
    #     loss = F.binary_cross_entropy_with_logits(output, target)
    #     score = self.metric(output, target)
    #     logs = {'train_loss': loss, 'train_f1': score, 'lr': self.optimizer.param_groups[0]['lr']}
    #     self.log_dict(
    #         logs,
    #         on_step=False, on_epoch=True, prog_bar=True, logger=True
    #     )
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     image = batch['image']
    #     target = batch['target']
    #     output = self.model(image)
    #     #output = self.sigmoid(output)
    #     #loss = self.criterion(output, target)
    #     loss = F.binary_cross_entropy_with_logits(output, target)
    #     score = self.metric(output, target)
    #     logs = {'valid_loss': loss, 'valid_f1': score}
    #     self.log_dict(
    #         logs,
    #         on_step=False, on_epoch=True, prog_bar=True, logger=True
    #     )
    #     return loss
