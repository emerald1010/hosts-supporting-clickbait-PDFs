import torch
from torchvision import models

from Models.Classification.BaseClassification import BaseClassificationModel


class Resnet18Model(BaseClassificationModel):

    def __init__(self):
        super(Resnet18Model, self).__init__(1)
        self.resnet_ft = models.resnet18(pretrained=False, num_classes=self.num_classes)


    def forward(self, x, **kwargs):
        x = self.resnet_ft(x)
        return torch.squeeze(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)
