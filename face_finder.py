#!/usr/bin/env python3

import os
from sklearn.model_selection import train_test_split
import glob
import numpy as np


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def all_file_paths() -> list[str]:

    base_path: str = 'Face Markup AR Database/points_22'
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    base_path: str = os.path.join(script_dir, base_path)

    if not os.path.exists(base_path):
        raise FileNotFoundError('Could not find the dataset folder')

    # Collect all file paths
    all_files: list[str] = glob.glob(os.path.join(base_path, 'm-*', '*.pts')) + \
        glob.glob(os.path.join(base_path, 'w-*', '*.pts'))

    return all_files


def filter_data(paths: list[str]) -> list[tuple[float, float]]:
    filtered_data: list[tuple[float, float]] = []
    for path in paths:
        with open(path) as file:
            lines = file.readlines()
            for line in lines[3:25]:
                x, y = line.split()
                filtered_data.append((float(x), float(y)))
    return filtered_data


def main():
    if __name__ == '__main__':
        file_paths: list = all_file_paths()
        data: list[tuple[float, float]] = filter_data(file_paths[:1])
        print(data)


main()
