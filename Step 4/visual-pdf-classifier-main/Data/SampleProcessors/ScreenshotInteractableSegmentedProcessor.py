import numpy as np
import torch
from PIL import Image
from numpy import asarray

from Data.SampleProcessors.BaseSampleProcessor import BaseSampleProcessor


class ScreenshotInteractableSegmentedProcessor(BaseSampleProcessor):

    def load_sample(self, sample, label):
        """
        :param sample: list
                    List containing endpoints to load the data belonging to a specific sample
        :param label: string or int
                    Label of the class the sample belongs to
        :return: np.Array
                    Numpy array containing the screenshot of the sample
        """

        screenshot = super().load_sample(sample, label)

        interactable_mask_path = sample[1]
        interactable_mask = self._apply_transformation(np.asarray(Image.open(interactable_mask_path)) / 255).int()

        formatted_sample = screenshot * interactable_mask
        return formatted_sample
