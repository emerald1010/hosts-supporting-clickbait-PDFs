import numpy as np
import torch
from PIL import Image
from numpy import asarray

from Data.SampleProcessors.BaseSampleProcessor import BaseSampleProcessor


class ScreenshotInteractableProcessor(BaseSampleProcessor):

    def load_sample(self, sample, label):
        """
        :param sample: list
                    List containing endpoints to load the data belonging to a specific sample
        :param label: string or int
                    Label of the class the sample belongs to
        :return: np.Array
                    Numpy array containing the screenshot of the sample
        """

        screenshot = torch.squeeze(super().load_sample(sample, label))

        interactable_mask_path = sample[1]
        interactable_mask = torch.squeeze(self._apply_transformation(np.asarray(Image.open(interactable_mask_path)) / 255))

        formatted_sample = torch.zeros((2, interactable_mask.shape[0], interactable_mask.shape[1]))
        formatted_sample[0, :, :] = screenshot
        formatted_sample[1, :, :] = interactable_mask

        return formatted_sample

    def process_sample(self, sample, label):
        """
        :param sample: list
                    List containing endpoints to load the data belonging to a specific sample
        :param label: string or int
                    Label of the class the sample belongs to

        :return: Sample ready to be feeded into a model
        """
        screenshot = self.load_sample(sample, label)

        return self.augmentation_transformations(screenshot)
