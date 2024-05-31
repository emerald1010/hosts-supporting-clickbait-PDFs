import random
from os import listdir
from os.path import join, isfile

from Data.Datalake.DataLake import DataLake


class ContagioDataLake(DataLake):
    """
        Class to load efficiently data from the Contagio dataset
    """

    def __init__(self, root_folder, benign_folder_name, malicious_folder_name):
        """

        Args:
            root_folder: folder where the data are located
            benign_folder_name: name of the folder (located in the root directory) containing benign files
            malicious_folder_name: name of the folder (located in the root directory) containing malicious files
        """
        super().__init__(root_folder)

        # save the path to the folder containing benign files
        self.benign_folder_path = join(self.root_folder, benign_folder_name)

        # save the path to the folder containing malicious files
        self.malicious_folder_path = join(self.root_folder, malicious_folder_name)

    def get_benign_samples_list(self, random_sampling: bool = False, count: int = False):
        """
        Get list of paths to benign samples

        Args:
            random_sampling: (default False)  sample random elements from the dataset
            count: (default False) maximum amount of elements to return

        Returns: list of str each one of whom is a path to a benign sample
        """

        # Load the files from the folder
        files = [(join(self.benign_folder_path, f),"generic benign") for f in listdir(self.benign_folder_path) if
                 (isfile(join(self.benign_folder_path, f)) and str(f).endswith('.pdf'))]

        # Shuffle the list
        if random_sampling:
            random.shuffle(files)

        # Return only the desired ammount of samples
        if not count or count > len(files):
            count = len(files)

        return files[:count]

    def get_malicious_samples_list(self, random_sampling=False, count: int = False):
        """
        Get list of paths to malicious samples

        Args:
            random_sampling: (bool,default False)  sample random elements from the dataset
            count: (int,default False) maximum amount of elements to return

        Returns: list of str each one of whom is a path to a malicious sample
        """

        # Load the files from the folder
        files = [(join(self.malicious_folder_path, f), "generic malicious") for f in listdir(self.malicious_folder_path) if
                 (isfile(join(self.malicious_folder_path, f)) and str(f).endswith('.pdf'))]

        # Shuffle the list
        if random_sampling:
            random.shuffle(files)

        # Return only the desired amount of samples
        if not count or count > len(files):
            count = len(files)

        return files[:count]

    def __len__(self):
        all_files = self.get_benign_samples_list() + self.get_malicious_samples_list()
        return len(all_files)


def prepare_contagio_datalake(config):
    """
    Use this function to prepare the Contagio dataset class given a configuration file
    Args:
        conf: dictionary containing the content of the configuration file

    Returns: instance of the ContagioDatalake class

    """

    # get from the config the path to the files of the contagio dataset
    contagio_entrypoint = config['global']['providers']['contagio']['path']

    # get from the config the name of the folders where the conta
    malpdf_folder = config['global']['providers']['contagio']['malicious_folder']
    benpdf_folder = config['global']['providers']['contagio']['benign_folder']

    # instantiate the ContagioDataset class
    datalake = ContagioDataLake(contagio_entrypoint, benpdf_folder, malpdf_folder)

    return datalake
