#!/usr/bin/env python3

"""
Data Processor for Face Recognition

This script contains the DataProcessor class 
which is used to process and extract features.
"""

import os
import glob
import numpy as np
import random
from typing import List, Dict
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self):
        self.training_features = []
        self.training_labels = []
        self.testing_features = []
        self.testing_labels = []

    @staticmethod
    def euclidean_distance(p1: float, p2: float) -> float:
        """
        Calculates the Euclidean distance between two points.

        Args:
            p1 (float): The first point.
            p2 (float): The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def all_file_paths(self) -> Dict[str, List[str]]:
        """
        Retrieves a dictionary of file paths for directories in the specified dataset folder
        that contain 4 or more .pts files.

        Returns:
            Dict[str, List[str]]: A dictionary where each key is a directory name and the value
            is a list of file paths for the .pts files in that directory.

        Raises:
            FileNotFoundError: If the specified dataset folder does not exist.
        """
        base_path = 'AR_DB/points_22'
        base_folder = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(base_folder, base_path)

        if not os.path.exists(base_path):
            raise FileNotFoundError('Could not find the dataset folder')

        # Collect all file paths grouped by their parent directory
        all_files = {}
        for gender_prefix in ['m', 'w']:
            dir_paths = glob.glob(os.path.join(
                base_path, f'{gender_prefix}-*'))
            for dir_path in dir_paths:
                files = glob.glob(os.path.join(dir_path, '*.pts'))
                if len(files) >= 4:
                    dir_name = os.path.basename(dir_path)
                    all_files[dir_name] = files

        return all_files

    @staticmethod
    def filter_data(paths: list[str]) -> list[tuple[str, float, float]]:
        """
        Extracts and filters data from files specified by the given paths.

        Args:
            paths (list[str]): A list of file paths from which to extract data.

        Returns:
            list[tuple[str, str, float]]: A list of tuples containing filtered data points.
        """
        filtered_data: List[tuple[str, float, float]] = []
        for path in paths:
            dir_name = os.path.basename(os.path.dirname(path))
            file_name = os.path.basename(path)
            points = []

            with open(path, 'r') as file:
                lines = file.readlines()
                for line in lines[3:len(lines) - 1]:
                    x, y = line.split()
                    points.append((float(x), float(y)))

            filtered_data.append((dir_name, file_name, points))

        return filtered_data

    def get_features(self, points: list[tuple[float, float]]) -> list[float]:
        """
        Extracts features from a list of points.

        Args:
            points (List[tuple[float, float]]): A list of (x, y) coordinates.

        Returns:
            List[float]: A list of extracted feature values.
        """

        # Extract features from data pointseye_length_ratio,
        # eye_distance_ratio, nose_ratio,lip_size_ratio,
        # lip_length_ratio, eyebrow_length_ratio, aggressive_ratio

        points = np.array(points)
        features = [
            self.euclidean_distance(
                points[9], points[10]) / self.euclidean_distance(points[8], points[13]),
            self.euclidean_distance(
                points[0], points[1]) / self.euclidean_distance(points[8], points[13]),
            self.euclidean_distance(
                points[15], points[16]) / self.euclidean_distance(points[20], points[21]),
            self.euclidean_distance(
                points[2], points[3]) / self.euclidean_distance(points[17], points[18]),
            self.euclidean_distance(
                points[2], points[3]) / self.euclidean_distance(points[20], points[21]),
            max(self.euclidean_distance(points[4], points[5]), self.euclidean_distance(
                points[6], points[7])) / self.euclidean_distance(points[8], points[13]),
            self.euclidean_distance(
                points[10], points[19]) / self.euclidean_distance(points[20], points[21])
        ]
        return features

    def organize(self, training_features: List, training_labels: List,
                 testing_features: List, testing_labels: List,
                 file_dict: Dict[str, List[str]], encoded_labels: Dict[str, int]):
        """
        Organizes data into training and testing sets.

        Args:
            training_features (List): The list to store training features.
            training_labels (List): The list to store training labels.
            testing_features (List): The list to store testing features.
            testing_labels (List): The list to store testing labels.
            file_dict (Dict[str, List[str]]): A dictionary of file paths grouped by directory name.
            encoded_labels (Dict[str, int]): A dictionary mapping directory names to encoded labels.

        Returns:
            tuple: A tuple containing updated training_features, training_labels, testing_features, and testing_labels.
        """
        for dir_name, file_paths in file_dict.items():
            random.shuffle(file_paths)
            training_paths = file_paths[:3]
            testing_paths = file_paths[3:4]

            for path in training_paths:
                data = self.filter_data([path])
                for _, _, points in data:
                    training_features.append(self.get_features(points))
                    training_labels.append(encoded_labels[dir_name])

            for path in testing_paths:
                data = self.filter_data([path])
                for _, _, points in data:
                    testing_features.append(self.get_features(points))
                    testing_labels.append(encoded_labels[dir_name])

        training_features, testing_features = self.scale_features(
            training_features, testing_features)
        return training_features, training_labels, testing_features, testing_labels

    @staticmethod
    def scale_features(training_features, testing_features):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(training_features)
        test_features = scaler.transform(testing_features)
        return train_features, test_features
