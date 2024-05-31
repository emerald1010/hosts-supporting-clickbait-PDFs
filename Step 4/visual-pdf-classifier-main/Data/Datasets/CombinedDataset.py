import logging
import os
import random
from collections import defaultdict
from os.path import join
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from Data.Datasets.SimpleMonoclassDataset import SimpleMonoclassDataset, load_directory_dataset
from Data.SampleProcessors.BaseSampleProcessor import BaseSampleProcessor
from Utilities.Logger.Logger import Logger


class CombinedDataset(Logger, Dataset):
    """
    This class is used to load and manage data where samples belonging to the same category can come form multiple
    sources
    """

    def __init__(self, data: dict, sample_processor: BaseSampleProcessor, label_rule="maliciousness"):
        """
        :param data: a dict containing an association  <category>: <datasets>
        """
        self.dataset = data

        self.sample_processor = sample_processor

        if self.dataset is None:
            self.dataset = defaultdict(dict)

        assert (label_rule in ["maliciousness", "sub-category-name"])
        self.label_rule = label_rule

    def compute_splits(self, n_test_samples, n_validation_samples, min_training_elements=0, shuffle=True):
        """
        Split the dataset into 3 disjoint sets for training validation and test.
        If a class has less thant n_test samples + n_validation_samples use it just for testing

        :param n_test_samples: int > 0
            number of elements to reserve for testing foreach subclass
        :param n_validation_samples: int > 0
            number of elements to reserve for testing foreach subclass
        :param min_training_elements: int > 0:
            If after having removed the validation and test set the subclass dataset has less than min_training_elements
            reserve this subclass of elements for testing
        :param shuffle: boolean
            shuffle elements before computing the splits
        :return:
        """

        training_data = defaultdict(dict)
        validation_data = defaultdict(dict)
        test_data = defaultdict(dict)

        reserve_for_test_threshold = n_validation_samples + n_test_samples + min_training_elements

        # Foreach category of data
        for class_key, class_datasets in self.dataset.items():

            # Foreach dataset of a given category
            for category_name, dataset in class_datasets.items():

                # if the dataset has less than n_training_theshold it is used jsut for test
                if len(dataset) < reserve_for_test_threshold:
                    test_data[class_key][category_name] = SimpleMonoclassDataset(dataset.samples, dataset.label,
                                                                                 dataset.campaing_name,
                                                                                 dataset.sample_processor,
                                                                                 dataset.label_rule)
                    continue

                samples = dataset.samples

                if shuffle:
                    random.shuffle(samples)

                training_set, validation_set, test_set = train_val_test_split(samples, n_validation_samples,
                                                                              n_test_samples)

                for element in training_set:

                    if element in validation_set:
                        print(element)
                        raise ValueError

                    if element in test_set:
                        print(element)
                        raise ValueError

                training_data[class_key][category_name] = SimpleMonoclassDataset(training_set, dataset.label,
                                                                                 dataset.campaing_name,
                                                                                 dataset.sample_processor,
                                                                                 dataset.label_rule)

                validation_data[class_key][category_name] = SimpleMonoclassDataset(validation_set, dataset.label,
                                                                                   dataset.campaing_name,
                                                                                   dataset.sample_processor,
                                                                                   dataset.label_rule)

                test_data[class_key][category_name] = SimpleMonoclassDataset(test_set, dataset.label,
                                                                             dataset.campaing_name,
                                                                             dataset.sample_processor,
                                                                             dataset.label_rule)

        return CombinedDataset(training_data, self.sample_processor, self.label_rule), \
               CombinedDataset(validation_data, self.sample_processor, self.label_rule), \
               CombinedDataset(test_data, self.sample_processor, self.label_rule)

    def pop_classes(self, class_names):
        """
        Remove the desired classes from the dataset and returns a new dataset containing just that
        :param class_names: iterable containing the classes to remove
        :return: Combined Dataset
        """
        popped = {}
        for class_name in class_names:
            popped[class_name] = self.dataset[class_name]

        del self.dataset[class_name]

        return CombinedDataset(popped, self.sample_processor)

    def count_class(self, cat_name):
        """
        Returns the total amount of elements for a give class
        :return:
        """
        tot = 0

        for dataset in self.dataset[cat_name].values():
            tot += len(dataset)

        return tot

    @property
    def classes(self):
        return self.dataset.keys()

    @property
    def sub_categories(self):
        subcats = {}
        for class_key in self.classes:

            for subclass_key, subclass_dataset in self.dataset[class_key].items():

                if subclass_key in subcats.keys():
                    raise ValueError

                subcats[subclass_key] = subclass_dataset

        return subcats

    @property
    def samples(self):
        samples = []
        for subcat_dataset in self.sub_categories.values():
            samples += subcat_dataset.samples

        return samples

    def __len__(self):
        count = 0
        for class_key in self.classes:
            for d in self.dataset[class_key].values():
                count += len(d)
        return count

    def __getitem__(self, index) -> T_co:

        assert index < len(self)

        for class_key in self.classes:

            if index < self.count_class(class_key):

                for subclass, dataset in self.dataset[class_key].items():

                    if index < len(dataset):
                        sample, label = dataset[index]

                        return sample, label

                    index -= len(dataset)

            index -= self.count_class(class_key)

        raise ValueError

    def sample_indexed_of_subclass(self,target_subclass,n_element):
        """
        Given a sublass of elements returns n indexes corresponding to the position of these
        """

        start_index = 0

        for class_key in self.classes:

            if target_subclass in self.dataset[class_key].keys():

                for subclass, dataset in self.dataset[class_key].items():

                    if subclass == target_subclass:
                         return random.sample(range(start_index, start_index + len(dataset)), min(n_element,len(dataset)))
                    else:
                        start_index += len(dataset)

            start_index -= self.count_class(class_key)

        raise ValueError


    def get_raw_sample(self, index):
        """
        Return the information of the desired sample without loading it
        :param index: int
            index of the desired sample
        :return: list
            containing the informations on the desired sample
        """

        for class_key in self.classes:

            if index < self.count_class(class_key):

                for subclass, dataset in self.dataset[class_key].items():

                    if index < len(dataset):
                        return dataset.get_raw_sample(index)

                    index -= len(dataset)

            index -= self.count_class(class_key)

    def get_item_class_subclass(self, index) -> tuple:
        for class_key in self.classes:
            if index < self.count_class(class_key):

                for subclass_name, sub_class_dataset in self.dataset[class_key].items():
                    if index < len(sub_class_dataset):
                        r = sub_class_dataset[index]
                        return r, class_key, subclass_name, index

                    index -= len(sub_class_dataset)

            index -= self.count_class(class_key)

        raise ValueError("Sample not found")

    @property
    def weights(self):
        """
        Returns a list of weights that foreach sample in the dataset denote the probability of picking that sample.
        These weights are computed to pick sample faily with respect to their class and their sub category
        :return:
        """

        subclasses_count_vectors = {}

        weights_list = []

        tot_class_weights = sum([self.__len__() / self.count_class(class_key) for class_key in self.classes])

        for class_key in self.classes:

            class_weight = self.__len__() / (self.count_class(class_key) * tot_class_weights)

            n_categories = len(self.dataset[class_key])

            for category_key, dataset in self.dataset[class_key].items():
                # cat_weight = self.count_class(class_key)/(len(dataset) * tot_cat_weights)

                avg_len = self.count_class(class_key) / n_categories

                weights_list += [class_weight * (avg_len / len(dataset))] * len(dataset)

        assert self.__len__() == len(weights_list)
        return np.array(weights_list) / sum(weights_list)

    @property
    def class_independent_weights(self):
        """
        Returns a list of weights that foreach sample in the dataset denote the probability of picking that sample.
        These weights are computed to pick sample farily with respect to just their sub category
        :return:
        """

        weights_list = []

        for class_key in self.classes:

            for category_key, dataset in self.dataset[class_key].items():
                # cat_weight = self.count_class(class_key)/(len(dataset) * tot_cat_weights)
                weights_list += [1 / len(dataset)] * len(dataset)

        assert self.__len__() == len(weights_list)
        return np.array(weights_list) / sum(weights_list)

    @property
    def min_length_subclass(self):
        return min([len(d) for d in self.sub_categories.values()])

    def __str__(self):
        output = ""
        for class_key in self.classes:
            output += f"{str(class_key).upper()} DATASETS: {self.count_class(class_key)}\n "
            for subclass_name, sub_dataset in self.dataset[class_key].items():
                output += f"\n {subclass_name.lower()}: {len(sub_dataset)}"
            output += "\n"

        if output == "":
            output = "Empty"
        return output

    def combine(self, second_dataset):
        """
        Given a second CombinedDataset, returns a new instance of the class containing the merged data
        :param second_dataset: dataset to merge
        :return:
        """
        new_dataset = self.dataset.copy()

        for class_key in second_dataset.dataset.keys():

            for category_key in second_dataset.dataset[class_key].keys():

                if category_key in new_dataset[class_key].keys():

                    new_dataset[class_key][category_key].samples += second_dataset.dataset[class_key][
                        category_key].samples
                else:
                    new_dataset[class_key][category_key] = second_dataset.dataset[class_key][category_key]

        return CombinedDataset(new_dataset, self.sample_processor, self.label_rule)

    def clamp_subclasses(self, min_count=0, max_count=1000, shuffle=True):
        """
        Clam the subclasses of the dataset to be in the provided range of lengths
        :param min_count: min value a subclass must have to be included
        :param max_count: max value a subclass must have to be included
        :param shuffle: shuffle the elements before clamping
        :return: Combined Dataset instance
        """

        new_dataset = defaultdict(dict)
        for class_name in self.classes:

            for subclass_name in self.dataset[class_name].keys():

                sub_dataset = self.dataset[class_name][subclass_name]

                if len(sub_dataset) < min_count:
                    continue

                samples = None

                if shuffle:
                    samples = random.sample(sub_dataset.samples, min(max_count, len(sub_dataset.samples)))
                else:
                    samples = sub_dataset.samples[:min(max_count, len(sub_dataset.samples))]

                new_subclass_dataset = SimpleMonoclassDataset(samples, sub_dataset.label, sub_dataset.campaing_name,
                                                              sub_dataset.sample_processor, sub_dataset.label_rule)
                new_dataset[class_name][subclass_name] = new_subclass_dataset
        return CombinedDataset(new_dataset, self.sample_processor, self.label_rule)

    def __add__(self, o):
        return self.combine(o)

    @property
    def dict_categories_samples(self):
        d = dict()

        for class_key in self.classes:

            for subclass_name, sub_dataset in self.dataset[class_key].items():
                d[subclass_name] = sub_dataset.list_sample_names

        return d


def train_val_test_split(values: list, val_length, test_length):
    train_cap = len(values) - val_length - test_length

    random.shuffle(values)

    return values[:train_cap], \
           values[train_cap:train_cap + val_length], \
           values[train_cap + val_length:train_cap + val_length + test_length]


def dict_datasetst_to_dataloaders(dict_dataset, num_workers=2, batch_size=16):
    dict_dataloaders = {}
    for key, dataset in dict_dataset.items():

        if isinstance(dataset, dict):
            dict_dataloaders[key] = dict_datasetst_to_dataloaders(dataset)
            continue

        dict_dataloaders[key] = DataLoader(dataset, pin_memory=True, num_workers=num_workers, batch_size=batch_size)

    return dict_dataloaders


def count_leafs(dictionaty):
    i = 0
    for key, dataset in dictionaty.items():

        if isinstance(dataset, dict):
            i += count_leafs(dataset)
            continue

        i += 1
    return i


def load_hierarchical_dataset(root_folder, sample_processor, label_rule="maliciousness"):
    root_folder = os.path.expanduser(root_folder)
    assert (Path(root_folder).exists())

    benign_root_folder = join(root_folder, "benign")
    malicious_root_folder = join(root_folder, "malicious")
    unlabelled_root_folder = join(root_folder)

    datasets = defaultdict(dict)
    if Path(benign_root_folder).exists():
        # load benign categories
        for folder in os.listdir(benign_root_folder):
            folder_path = join(benign_root_folder, folder)
            if os.path.isdir(folder_path):
                try:
                    dataset = load_directory_dataset(folder_path, 0, sample_processor, label_rule)
                    datasets["benign"][folder] = dataset
                except Exception as e:
                    print(e)
                    logging.warning(f"Could not load dataset for benign class {folder}")
                    continue

    if Path(malicious_root_folder).exists():

        # load malicious categories
        for folder in os.listdir(malicious_root_folder):

            folder_path = join(malicious_root_folder, folder)

            if os.path.isdir(folder_path):

                try:
                    dataset = load_directory_dataset(folder_path, 1, sample_processor, label_rule)

                    if dataset:
                        datasets["malicious"][folder] = dataset
                except Exception as e:
                    logging.warning(f"Could not load dataset for malicious class {folder}")
                    continue


    for folder in os.listdir(unlabelled_root_folder):

        if folder not in ["benign", "malicious" ]:

            folder_path = join(unlabelled_root_folder, folder)

            if os.path.isdir(folder_path):

                try:
                    dataset = load_directory_dataset(folder_path, -1, sample_processor, label_rule)

                    if dataset:
                        datasets["unlabelled"][folder] = dataset

                except Exception as e:
                    logging.warning(f"Could not load dataset for unlabelled class {folder}")
                    continue

    dataset = CombinedDataset(datasets, sample_processor, label_rule)
    return dataset
