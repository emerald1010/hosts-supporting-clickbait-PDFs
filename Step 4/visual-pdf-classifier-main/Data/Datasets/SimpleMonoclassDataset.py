import logging
import os
import random
from os.path import join, basename
from pathlib import Path

import torch
from numpy import asarray
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms, RandomHorizontalFlip, RandomVerticalFlip

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.ERROR)
from Utilities.Logger.Logger import Logger


class SimpleMonoclassDataset(Dataset, Logger):
    """
    This class contains the smallest and siplest possible implementation of a Datset class containing
    only samples of the same class
    """

    def __init__(self, samples, label = -1,campaing_name = "unknown", sample_processor=None,label_rule="maliciousness"):
        self.samples = samples
        self.label = label
        self.campaing_name = campaing_name
        self.sample_processor = sample_processor

        assert (label_rule in ["maliciousness", "sub-category-name"])
        self.label_rule = label_rule

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.sample_processor:
            sample = self.sample_processor.process_sample(sample,self.campaing_name)

        label = self.label

        if self.label_rule == "sub-category-name":
            label = self.campaing_name

        return sample, label

    def get_raw_sample(self, index):
        """
        Return the information of the desired sample without loading it
        :param index: int
            index of the desired sample
        :return: list
            containing the informations on the desired sample
        """
        sample = self.samples[index]

        label = self.label

        if self.label_rule == "sub-category-name":
            label = self.campaing_name

        return sample, label

    @property
    def random_sample(self):
        index = random.randrange(self.__len__())
        return self.__getitem__(index)

    def __len__(self):
        return len(self.samples)

    @property
    def list_sample_names(self):
        return [basename(s[0].split(".")[0]) for s in self.samples]

def load_directory_dataset(root_folder, label, sample_processor,label_rule):
    """
    Given a folder and a label, create a dataset containing the samples it that folder associated to the given label
    :return: SimpleMonoclassDataset
    """

    campaign_name = basename(root_folder)

    screenshots_folder = join(root_folder, "screenshots")

    if not Path(screenshots_folder).exists():
        screenshots_folder = root_folder

    interactable_masks_first_page_folder = join(root_folder, "interactive-masks-first-page")

    samples = []

    print(screenshots_folder)

    for file_name in os.listdir(screenshots_folder):

        if file_name.endswith(".png"):

            screenshot_path = join(screenshots_folder, file_name)

            first_page_interactable_mask_path = join(interactable_masks_first_page_folder, file_name)

            samples.append((screenshot_path, first_page_interactable_mask_path))

    if not samples:
        return False

    return SimpleMonoclassDataset(samples, label,campaign_name, sample_processor,label_rule)
