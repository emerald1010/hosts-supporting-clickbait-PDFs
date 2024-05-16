from abc import ABC, abstractmethod


class DataLake(ABC):
    """
        Dataset base class
    """

    @staticmethod
    def prepare_datalake(config):
        """
        Function that returns an instance of the DATALAKE class instantiated using the right parameters
        specified in the configs.
        :param config: instance of the Config class
        :return: instance of the DATALAKE class
        """
        raise NotImplementedError
