#!/usr/bin/env python3

"""
@author: Wilkenson | Kyle | Aeron
@date: 01/01/2024  
"""

import os
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

# euclidean_distance function


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

# Glob files into one directory


def all_file_paths() -> Dict[str, List[str]]:
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
        dir_paths = glob.glob(os.path.join(base_path, f'{gender_prefix}-*'))
        for dir_path in dir_paths:
            files = glob.glob(os.path.join(dir_path, '*.pts'))
            if len(files) >= 4:
                dir_name = os.path.basename(dir_path)
                all_files[dir_name] = files

    return all_files


# Example usage
directories_with_files = all_file_paths()
# print(directories_with_files)
# Filter data from glob directoy into a list of tuples


def filter_data(paths: list[str]) -> list[tuple[float, float]]:
    """
    Extracts and filters data from files specified by the given paths.

    Args:
        paths (list[str]): A list of file paths from which to extract data.

    Returns:
        list[tuple[float, float]]: A list of tuples containing filtered data points.
    """

    filtered_data = []
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


# Extract features from data points
def get_features(points: list[tuple[float, float]]) -> list[float]:
    """
    Extracts features from a list of points.

    Args:
        points (List[tuple[float, float]]): A list of (x, y) coordinates.

    Returns:
        List[float]: A list of extracted feature values.
    """
    features = []
    points = np.array(points)
    # 7 features needed for the model need to verify the number values in this code block
    eye_length_ratio = euclidean_distance(
        points[9], points[10]) / euclidean_distance(points[8], points[13])
    eye_distance_ratio = euclidean_distance(
        points[0], points[1]) / euclidean_distance(points[8], points[13])
    nose_ratio = euclidean_distance(
        points[15], points[16]) / euclidean_distance(points[20], points[21])
    lip_size_ratio = euclidean_distance(
        points[2], points[3]) / euclidean_distance(points[17], points[18])
    lip_length_ratio = euclidean_distance(
        points[2], points[3]) / euclidean_distance(points[20], points[21])
    eyebrow_length_ratio = max(euclidean_distance(points[4], points[5]), euclidean_distance(
        points[6], points[7])) / euclidean_distance(points[8], points[13])
    aggressive_ratio = euclidean_distance(
        points[10], points[19]) / euclidean_distance(points[20], points[21])
    # Appending features to a list
    features.extend([eye_length_ratio,
                     eye_distance_ratio,
                     nose_ratio,
                     lip_size_ratio,
                     lip_length_ratio,
                     eyebrow_length_ratio,
                     aggressive_ratio
                     ])
    return features

# Organize the data into training and testing data


def organize(training_features: List, training_labels: List,
             testing_features: List, testing_labels: List,
             file_dict: Dict[str, List[str]], encoded_labels: Dict[str, int]
             ):
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

    # Iterate over each directory's files
    for dir_name, file_paths in file_dict.items():
        # Shuffle the file paths to ensure randomness
        random.shuffle(file_paths)
        # Splitting the data: first 3 for training, last one for testing
        training_paths = file_paths[:3]  # First three files for training
        testing_paths = file_paths[3:4]  # Only one file for testing

        # Process training data
        for path in training_paths:
            data = filter_data([path])  # filter_data expects a list
            for _, file_name, points in data:
                training_features.append(get_features(points))
                training_labels.append(encoded_labels[dir_name])

        # Process testing data
        for path in testing_paths:
            data = filter_data([path])  # filter_data expects a list
            for _, file_name, points in data:
                testing_features.append(get_features(points))
                testing_labels.append(encoded_labels[dir_name])
    training_features, testing_features = scale_features(
        training_features, testing_features)
    return training_features, training_labels, testing_features, testing_labels


def scale_features(training_features, testing_features):
    scaler = StandardScaler()
    train_features = scaler.fit_transform(training_features)
    test_features = scaler.transform(testing_features)
    return train_features, test_features


def knn(training_features, training_labels, testing_features, testing_labels):
    """
    Trains and evaluates a k-Nearest Neighbors (kNN) classifier.

    Args:
        training_features (List): Training feature vectors.
        training_labels (List): Training labels.
        testing_features (List): Testing feature vectors.
        testing_labels (List): Testing labels.
    """
    train_features = np.array(training_features)
    train_labels = np.array(training_labels)
    test_features = np.array(testing_features)
    test_labels = np.array(testing_labels)

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(train_features, train_labels)

    # Step 4: Test the model
    predictions = knn.predict(test_features)

    # Unique labels in predictions
    unique_predicted_labels = set(predictions)
    # print("Unique Predicted Labels:", unique_predicted_labels)

    # Unique labels in the test set
    unique_test_labels = set(test_labels)
    # print("Unique Test Labels:", unique_test_labels)

    # Evaluate the model
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report: (KNN)")
    print(classification_report(test_labels, predictions, zero_division=0))


def naives_bayes(training_features, training_labels, testing_features, testing_labels) -> None:
    """
    Trains and evaluates a Gaussian Naive Bayes classifier.

    Args:
        training_features (List): Training feature vectors.
        training_labels (List): Training labels.
        testing_features (List): Testing feature vectors.
        testing_labels (List): Testing labels.
    """

    gnd = GaussianNB()
    gnd.fit(training_features, training_labels)  # Training the model
    predictions = gnd.predict(testing_features)

    accuracy = accuracy_score(testing_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report: (naives bayes)")
    print(classification_report(testing_labels, predictions, zero_division=0))


def ann(training_features, training_labels, testing_features, testing_labels):
    ann_classifier = MLPClassifier(hidden_layer_sizes=(
        100,), max_iter=10000, activation='relu', solver='adam', random_state=1)
    ann_classifier.fit(training_features, training_labels)
    ann_predictions = ann_classifier.predict(testing_features)
    accuracy = accuracy_score(testing_labels, ann_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report: (ANN)")
    print(classification_report(testing_labels, ann_predictions, zero_division=0))


def svm(training_features, training_labels, testing_features, testing_labels):
    # RBF kernel is sensitive to Euclidean distances between points
    svm_classifier = SVC(kernel='rbf', C=1)
    svm_classifier.fit(training_features, training_labels)
    svm_predictions = svm_classifier.predict(testing_features)
    accuracy = accuracy_score(testing_labels, svm_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report: (SVM)")
    print(classification_report(testing_labels, svm_predictions, zero_division=0))


def decision_tree(training_features, training_labels, testing_features, testing_labels):
    pass


def display_report(file_dict: dict[str, str], encoded_labels: list[int]) -> None:
    """
    Displays the classification report for the model.
    """
    training_features = []
    training_labels = []
    testing_features = []
    testing_labels = []
    training_features, training_labels, testing_features, testing_labels = organize(
        training_features, training_labels,
        testing_features, testing_labels, file_dict, encoded_labels)

    # knn(training_features, training_labels, testing_features, testing_labels)
    # print("------------------------------------------------------")
    # naives_bayes(training_features, training_labels,
    #              testing_features, testing_labels)
    # print("------------------------------------------------------")
    ann(training_features, training_labels, testing_features, testing_labels)
    print("------------------------------------------------------")
    svm(training_features, training_labels, testing_features, testing_labels)
    print("------------------------------------------------------")
    # decision_tree(training_features, training_labels,
    #               testing_features, testing_labels)
    # print("------------------------------------------------------")


def main():
    file_dict = all_file_paths()
    # List of labels in txt form before we encode them
    all_labels = list(file_dict.keys())

    label_encoder = LabelEncoder()

    label_encoder.fit_transform(all_labels)  # Encoding the labels

    # encoded_labels dict['m-001': 0], filenme=m-001 and index=0 where index represents the class
    encoded_labels = {dir_name: label_encoder.transform(
        [dir_name])[0] for dir_name in file_dict.keys()}

    display_report(file_dict, encoded_labels)


if __name__ == '__main__':
    main()
