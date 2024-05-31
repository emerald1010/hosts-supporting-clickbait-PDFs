class BaseFile:
    """
    This files defin the structure common to all the FileTypes
    """

    def __init__(self,path):
        """
        :param path: the path to the file
        """
        self.path = path