#!/usr/bin/env python3

"""
@author: Wilkenson | Kyle | Aeron
@date: 01/01/2024  
"""

import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import glob
import numpy as np

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
def all_file_paths() -> list[str]:
    """
    Retrieves a list of file paths for files in the specified dataset folder.

    Returns:
        list[str]: A list of file paths for the dataset files.

    Raises:
        FileNotFoundError: If the specified dataset folder does not exist.
    """

    base_path: str = 'AR_DB/points_22'
    base_folder: str = os.path.dirname(os.path.abspath(__file__))
    base_path: str = os.path.join(base_folder, base_path)

    if not os.path.exists(base_path):
        raise FileNotFoundError('Could not find the dataset folder')

    # Collect all file paths
    all_files: list[str] = glob.glob(os.path.join(base_path, 'm-*', '*.pts')) + \
        glob.glob(os.path.join(base_path, 'w-*', '*.pts'))
    all_files.sort()
    return all_files

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
        
    print ("dir_name, file_name, points\n", filtered_data, "\n")
                
                
    
    return filtered_data

# Get features from the filtered data putting them into a list based on the points array
def get_features(points: list[tuple[float, float]]) -> list[float]:
    
    features = []
    points = np.array(points)
    # 7 features needed for the model need to verify the number values in this code block
    eye_length_ratio = euclidean_distance(points[9], points[10]) / euclidean_distance(points[8], points[13])
    eye_distance_ratio = euclidean_distance(points[1], points[5]) / euclidean_distance(points[7], points[12])
    nose_ratio = euclidean_distance(points[14], points[15]) / euclidean_distance(points[19], points[20])
    lip_size_ratio = euclidean_distance(points[1], points[2]) / euclidean_distance(points[16], points[17])
    lip_length_ratio = euclidean_distance(points[1], points[2]) / euclidean_distance(points[19], points[20])
    eyebrow_length_ratio = max(euclidean_distance(points[3], points[4]), euclidean_distance(points[5], points[6])) / euclidean_distance(points[7], points[12])
    aggressive_ratio = euclidean_distance(points[9], points[18]) / euclidean_distance(points[19], points[20])
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


def main():
    file_paths = all_file_paths()
    data = filter_data(file_paths)  # each tuple is (dir_name, file_name, points)
    
    # Splitting dataset into features and labels for training/testing
    features = []
    labels = []
    

    for dir_name, file_name, points in data:
        features.append(get_features(points))
        labels.append(dir_name)
        
    print("Features:",features,"/n","labels", labels,"/n")

    features = np.array(features)
    labels = np.array(labels)

    #Boiler plate code for splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # print(f"Training set size: {len(X_train)}")
    # print(f"Testing set size: {len(X_test)}")

    # # sample 
    # print("Sample training data:", X_train[:5])
    # print("Sample testing data:", X_test[:5])

if __name__ == '__main__':
    main()


