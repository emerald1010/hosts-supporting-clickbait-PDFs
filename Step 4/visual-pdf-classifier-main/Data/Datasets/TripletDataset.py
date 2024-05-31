import random
from Data.Datasets.CombinedDataset import CombinedDataset, load_hierarchical_dataset
from Data.SampleProcessors.BaseSampleProcessor import BaseSampleProcessor


class TripletDataset(CombinedDataset):
    """
    This class implements a dataset returning:
        anchor,positive,negative samples
    It is useful to train a Siamese Network
    """

    def __init__(self, data,sample_processor: BaseSampleProcessor):
        """
        :param data:
        """
        super().__init__(data,sample_processor)

    def compute_splits(self, validation_split=0.1, test_split=0.1, subclass_val_length=None, n_training_threshold=1000,
                       limit_training_elements=None):
        """
        Generate a training,validation.test datasets by splitting elements randomly
        :param validation_split: we have more than n_training_theshold in a category, how much of them should we save for
        validation purposes?
        :param test_slit: we have more than n_training_theshold in a category, how much of them should we save for
        testing purposes?
        :param limit_training_elements: the ammounts of training elements per class that should be used
        :return: training dataset, validation dataset, [test datasets]
        """

        train_split, val_split, test_split = super(TripletDataset, self).compute_splits(validation_split, test_split,
                                                                                        subclass_val_length,
                                                                                        n_training_threshold,
                                                                                        limit_training_elements)

        return TripletDataset(train_split.dataset), TripletDataset(val_split.dataset), TripletDataset(
            test_split.dataset)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        anchor, anchor_class_name, anchor_subclass_name, anchor_subclass_index = self.get_item_class_subclass(item)

        # sample example from the same class as the anchor
        positive = None
        while True:
            positive_index = random.randrange(0, len(self.dataset[anchor_class_name][anchor_subclass_name]))

            # be sure anchor and positive are different samples
            if positive_index != anchor:
                positive = self.dataset[anchor_class_name][anchor_subclass_name][positive_index]
                break

        # sample example from a different class than the anchor
        while True:
            negative_class = random.choice(list(self.dataset.keys()))
            negative_subcategory = random.choice(list(self.dataset[negative_class].keys()))

            if negative_class != anchor_class_name or negative_subcategory != anchor_subclass_name:
                break

        negative = self.dataset[negative_class][negative_subcategory].random_sample

        # return anchor, positive, negative samples
        return anchor[0], positive[0], negative[0]

def load_triplet_dataset(root_folder):
    combined_dataset = load_hierarchical_dataset(root_folder)

    return TripletDataset(combined_dataset.dataset)
