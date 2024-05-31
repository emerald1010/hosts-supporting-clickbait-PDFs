from abc import ABC
from Utilities.Logger.Logger import Logger
import pytorch_lightning as pl


class BaseModel(pl.LightningModule, Logger, ABC):

    def __init__(self):
        super().__init__()