import torch.nn as nn
import torch


from Utilities.Logger.Logger import Logger


class BasePytorchModel(nn.Module,Logger):
    """
    This class defines an interface for all the other pytorch models to use
    """

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path):
        torch.load(path)
