import os
import shutil
from pathlib import Path


def _rel_dest_path(fname, nested_levels=4):
    """
    E.g., if fname = "abcdefghilmnopq", this function returns "ab/cd/ef/gh/"
    """
    if len(fname) >= 2 * nested_levels:
        return "/".join([fname[2 * i:2 * (i + 1)] for i in range(0, nested_levels)]) + "/"
    raise Exception(
        "Relative file storage path cannot be determined because len({})={} < {} chars".format(fname, len(fname),
                                                                                               2 * nested_levels))


def filename_path_decode(basename, root_destination_folder):
    """
    Given a filename and a root folder this function returns the multilevel path inside the root of the folder in which
    the sample has to be saved
    :param basename: name of the file without extension
    :param root_destination_folder: root folder in which to save the data
    :return: str
    """
    rel_dest_path = _rel_dest_path(basename)
    dest_path = os.path.join(root_destination_folder, rel_dest_path)
    return dest_path


def add_sample(sample_path, root_path):
    """
    This function takes a sample path and moves it into the specified root folder creating also a set of 4 nested folders
    whose names depend on the filename of the specified sample
    :param sample_path: path of the sample to move
    :param root_path: root folder
    :return: bool
    """
    # Make sure that sample_path corresponds to an existing file
    assert Path(sample_path).exists()

    # Get the pure filename of the file without extension
    basename = os.path.splitext(os.path.basename(sample_path))[0]

    # Get the path of the destination folder in which to save the file
    dest_folder_path = filename_path_decode(basename, root_path)

    # If the destination folder does not exist, create it
    if not Path(dest_folder_path).exists():
        os.makedirs(dest_folder_path)

    # Compute the final path of the file
    dest_file_path = os.path.join(dest_folder_path, basename)

    # Make sure that no file already exists at this path
    assert not Path(dest_file_path).exists()

    # Copy the sample in the destination path
    shutil.copy(sample_path, dest_file_path)
