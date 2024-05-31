from abc import ABC
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from Models.Classification.BaseModel import BaseModel


class BaseClassificationModel(BaseModel, ABC):

    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.training_loss = nn.BCEWithLogitsLoss()
        self.training_acc = Accuracy()

        self.val_loss = nn.BCEWithLogitsLoss()
        self.valid_acc = Accuracy()

        self.test_acc = Accuracy()
        self.subcat_name = ""
        self.test_dataloaders_list = None

        self.test_accu_dict = defaultdict(lambda: defaultdict(Accuracy))

    def training_step(self, batch, batch_idx):
        self.training = True
        x, y = batch
        y_hat = self.forward(x)
        loss_value = self.training_loss(y_hat, y.float())
        self.training_acc(torch.round(y_hat), y)
        self.log('training_accuracy', self.training_acc, on_step=True, on_epoch=True)
        return {"loss": loss_value}

    def validation_step(self, batch, batch_idx):
        self.training = False
        x, y = batch
        y_hat = self.forward(x)
        loss_value = self.val_loss(y_hat, y.float())
        self.valid_acc(torch.round(y_hat), y)
        self.log('validation_epoch_accuracy', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss_value}

    def test_step(self, batch, batch_idx, dataset_idx=0):
        self.training = False
        x, y = batch
        y_hat = self.forward(x)
        self.test_acc(torch.round(y_hat), y)
        self.log(f'{self.subcat_name}_test_acc', self.test_acc, on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs):
        self.logger_module.info(f" TRAINING ACC: {self.training_acc.compute()}")

    def validation_epoch_end(self, outputs):
        self.logger_module.info(f" VALIDATION ACC: {self.valid_acc.compute()}")

    def test_epoch_end(self, outputs):
        self.logger_module.info(f"TEST ACC {self.subcat_name}: {self.test_acc.compute()}")

    def setup_test(self, subcat_name):
        self.subcat_name = subcat_name
        self.test_acc.reset()
