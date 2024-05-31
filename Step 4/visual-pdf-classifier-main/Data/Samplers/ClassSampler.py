import random

from torch.utils.data.sampler import Sampler

from Data.Datasets.CombinedDataset import CombinedDataset


class SubclassSampler(Sampler):

    def __init__(self, dataset: CombinedDataset, batch_size, classes_in_batch):
        super().__init__(None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.classes_in_batch = classes_in_batch

    def __iter__(self):
        for i in range(len(self)):
            picked_subclasses = random.sample(list(self.dataset.sub_categories), self.classes_in_batch)
            samples_indexes = []

            for subclass in picked_subclasses:
                samples_indexes += self.dataset.sample_indexed_of_subclass(subclass,
                                                                            self.batch_size // self.classes_in_batch)

            random.shuffle(samples_indexes)

            yield samples_indexes
    def __len__(self):
        return len(self.dataset) // self.batch_size
