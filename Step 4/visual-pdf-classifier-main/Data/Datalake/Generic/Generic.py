import glob
import os
from os import listdir
from pathlib import Path
from os.path import join, isfile, isdir

from Data.Datalake.DataLake import DataLake


class GenericDatalake(DataLake):
    """
    This class manages generic datatalakes whose files are divided in different folders
    each one signaling a different category
    """

    def __init__(self, root_folder, benign_subfolder_name = "benign", malicious_subfolder_name = "malicious"):
        """
        Args:
            root_folder: folder where the data are located
            positives_categories: list of positive categories
            malicious_categories: list of negative categories
        """
        super().__init__(root_folder)

    def get_benign_categories_list(self, min_count=None):
        """
        Get the list of malicious categories names we can use to filter the files

        Args:
            min_count: minimum amount of elements a category must have to be returned
        Returns: list of strings
        """

        for cat_name in self.benign_categories_list:
            files_in_cat = glob.glob(join(self.root_folder, cat_name, "*.pdf"))

            if min_count and len(files_in_cat) < min_count:
                continue

            yield cat_name, len(files_in_cat)

    def get_malicious_categories_list(self, min_count=None):
        for cat_name in self.malicious_categories_list:
            files_in_cat = glob.glob(join(self.root_folder, cat_name, "*.pdf"))

            if min_count and len(files_in_cat) < min_count:
                continue

            yield cat_name, len(files_in_cat)

    def get_benign_samples_list(self, random_sampling: bool = False, count: int = False,white_list_categories=None):

        files = []
        for positive_category in self.benign_categories_list:

            if white_list_categories and positive_category not in white_list_categories:
                continue

            category_folder = join(self.benign_root, positive_category)

            if not Path(category_folder):
                continue

            category_files = [(join(category_folder, f), positive_category) for f in listdir(category_folder) if
                              (isfile(join(category_folder, f)) and str(f).endswith('.pdf'))]

            files += category_files

        return files

    def get_malicious_samples_list(self, random_sampling=False, count: int = False,white_list_categories=None):
        files = []
        for malicious_category in self.malicious_categories_list:

            if white_list_categories and malicious_category not in white_list_categories:
                continue

            category_folder = join(self.malicious_root, malicious_category)

            if not Path(category_folder):
                continue

            category_files = [(join(category_folder, f), malicious_category) for f in listdir(category_folder) if
                              (isfile(join(category_folder, f)) and str(f).endswith('.pdf'))]

            files += category_files

        return files

    def __len__(self):
        return len(self.get_benign_samples_list() + self.get_malicious_samples_list())


    @property
    def benign_root(self):
        return join(self.root_folder,"benign")

    @property
    def malicious_root(self):
        return join(self.root_folder,"malicious")

    @property
    def benign_categories_list(self):
        return [cat_name for cat_name in listdir(self.benign_root) if
                         isdir(join(self.benign_root, cat_name))]

    @property
    def malicious_categories_list(self):
        return [cat_name for cat_name in listdir(self.malicious_root) if
         isdir(join(self.malicious_root, cat_name))]

def prepare_generic_datalake(entrypoint):
    """
    Use this function to prepare the generic datalake class given a configuration file
    Args:
        conf: dictionary containing the content of the configuration file

    Returns: instance of the ContagioDataset class

    """

    entrypoint = os.path.expanduser(entrypoint)
    # instantiate the ContagioDataset class
    datalake = GenericDatalake(entrypoint)

    return datalake
