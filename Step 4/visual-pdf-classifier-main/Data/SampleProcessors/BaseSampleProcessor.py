from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from Data.Augmentations.SaltAndPepper import SaltAndPepperNoise


class BaseSampleProcessor():

    def __init__(self, side_length: int = None, input_channels=3, scale_random_crop=(1, 1),
                 advanced_augmentations=True, augmentation_transformations_list=[]):

        self.transformations_list = [transforms.ToTensor()]
        self.augmentation_transformations_list = augmentation_transformations_list

        self.side_length = side_length
        if self.side_length is not None:
            self.augmentation_transformations_list.append(transforms.Resize((side_length,side_length)),)

        self.input_chennels = input_channels

        self.advanced_augmentations = advanced_augmentations
        if self.advanced_augmentations:
            self.augmentation_transformations_list.append(SaltAndPepperNoise(noiseType='SnP', imgType='PIL'))

        if self.input_chennels == 1:
            self.transformations_list.append(transforms.Grayscale(num_output_channels=1))

        self.augmentation_transformations = transforms.Compose(self.augmentation_transformations_list)

        self.transformations = transforms.Compose(self.transformations_list)
        

    def load_sample(self, sample, label=None):
        """
        :param sample: list
                    Endpoints to load the data belonging to a specific sample
        :param label: string or int
                    Label of the class the sample belongs to (added for consistency reasons)
        :return: np.Array
                    Numpy array containing the screenshot of the sample
        """
        try:
            screenshot_path = sample[0]
            screenshot = Image.open(screenshot_path)
        except:
            print(sample[0])
            raise Exception("Exception loading the screenshot")
        return screenshot

    def process_sample(self, sample, label=None):
        """
        :param sample: list
                    List containing endpoints to load the data belonging to a specific sample
        :param label: string or int

        :return: Sample ready to be feeded into a model
        """
        combined_sample = self.load_sample(sample, label)
        return self._apply_transformation(self.augmentation_transformations(combined_sample)).float()

    def _apply_transformation(self, sample):
        """
        Apply the base transformations to load a model corretly, ensuring that the output is a torch.Tensor
        with the most compact representation possible.
        :param sample:
        :return:
        """
        return self.transformations(sample)
