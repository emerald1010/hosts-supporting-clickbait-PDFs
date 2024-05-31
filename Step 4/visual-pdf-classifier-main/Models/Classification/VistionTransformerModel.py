import torch
import timm
from torch import nn

from Models.Classification.BaseClassification import BaseClassificationModel
from Models.Classification.VisionTransformer.Components.VisionTransformer import VisionTransformer


class VisionTransformerModel(BaseClassificationModel):

    def __init__(self, pretrained=True):
        super(VisionTransformerModel, self).__init__(1)

        self.vt = timm.create_model("vit_base_patch16_384", pretrained=pretrained, num_classes=1)

    def forward(self, x, **kwargs):
        x = self.vt(x)
        return torch.squeeze(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
