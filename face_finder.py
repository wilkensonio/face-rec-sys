#!/usr/bin/env python3

"""
@author: Wilkenson | Kyle | Aeron
@date: 01/01/2024  
"""

import os
from sklearn.model_selection import train_test_split
import glob
import numpy as np


def euclidean_distance(p1: float, p2: float) -> float:
    """
    Calulates the Euclidean distance between two points.

    Args:
        p1 (float): The first point.
        p2 (float): The second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def all_file_paths() -> list[str]:
    """
    Retrieves a list of file paths for files in the specified dataset folder.

    Returns:
        list[str]: A list of file paths for the dataset files.

    Raises:
        FileNotFoundError: If the specified dataset folder does not exist.
    """

    base_path: str = 'Face Markup AR Database/points_22'
    base_folder: str = os.path.dirname(os.path.abspath(__file__))
    base_path: str = os.path.join(base_folder, base_path)

    if not os.path.exists(base_path):
        raise FileNotFoundError('Could not find the dataset folder')

    # Collect all file paths
    all_files: list[str] = glob.glob(os.path.join(base_path, 'm-*', '*.pts')) + \
        glob.glob(os.path.join(base_path, 'w-*', '*.pts'))
    all_files.sort()
    return all_files


def filter_data(paths: list[str]) -> list[tuple[float, float]]:
    """
    Extracts and filters data from files specified by the given paths.

    Args:
        paths (list[str]): A list of file paths from which to extract data.

    Returns:
        list[tuple[float, float]]: A list of tuples containing filtered data points.
    """

    filtered_data: list[tuple[float, float]] = []

    for path in paths:
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines[3:len(lines) - 1]:
                x, y = line.split()
                filtered_data.append((float(x), float(y)))

    return filtered_data


def main():
    if __name__ == '__main__':
        file_paths: list[str] = all_file_paths()
        data: list[tuple[float, float]] = filter_data(
            file_paths)  # each tuple is a point


main()
