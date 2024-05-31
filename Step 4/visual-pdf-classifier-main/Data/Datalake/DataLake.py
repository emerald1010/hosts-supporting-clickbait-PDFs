from abc import ABC, abstractmethod


class DataLake(ABC):
    """
        Dataset base class
    """

    def __init__(self, root_folder):
        """
        Args:
            root_folder: folder where the data are located
        """
        self.root_folder = root_folder

    @abstractmethod
    def get_benign_samples_list(self, random_sampling: bool = False, count: int = False):
        """
        Get list of paths to benign samples

        Args:
            random_sampling: (default False)  sample random elements from the dataset
            count: (default False) maximum amount of elements to return

        Returns: list of str each one of whom is a path to a benign sample
        """

        raise NotImplementedError

    @abstractmethod
    def get_malicious_samples_list(self, random_sampling=False, count: int = False):
        """
        Get list of paths to malicious samples

        Args:
            random_sampling: (bool,default False)  sample random elements from the dataset
            count: (int,default False) maximum amount of elements to return

        Returns: list of str each one of whom is a path to a malicious sample
        """

        raise NotImplementedError

    def get_benign_categories_list(self, min_count = None):
        """
        Get the list of benign categories names we can use to filter the files

        Args:
            min_count: minimum amount of elements a category must have to be returned
        Returns: list of strings or None if the dataset does not make any distinction between files
        """
        raise NotImplementedError

    def get_malicious_categories_list(self, min_count = None):
        """
        Get the list of malicious categories names we can use to filter the files

        Args:
            min_count: minimum amount of elements a category must have to be returned
        Returns: list of strings or None if the dataset does not make any distinction between files

        """
        raise NotImplementedError

    def __len__(self):
        """
        How many samples (Pristine + Malicious) does this dataset has?
        Returns: int
        """
        raise NotImplementedError
