#!/usr/bin/env python3

"""
@author: Wilkenson | Kyle | Aeron
@date: 01/01/2024  
"""

import os
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets, metrics
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

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
print(directories_with_files)
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
    eye_distance_ratio = euclidean_distance(points[0], points[1]) / euclidean_distance(points[8], points[13])
    nose_ratio = euclidean_distance(points[15], points[16]) / euclidean_distance(points[20], points[21])
    lip_size_ratio = euclidean_distance(points[2], points[3]) / euclidean_distance(points[17], points[18])
    lip_length_ratio = euclidean_distance(points[2], points[3]) / euclidean_distance(points[20], points[21])
    eyebrow_length_ratio = max(euclidean_distance(points[4], points[5]), euclidean_distance(points[6], points[7])) / euclidean_distance(points[8], points[13])
    aggressive_ratio = euclidean_distance(points[10], points[19]) / euclidean_distance(points[20], points[21])
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
    file_dict = all_file_paths()
    all_data = []
    features = []
    labels = []

    # Iterate over each directory's files
    for dir_name, file_paths in file_dict.items():
        data = filter_data(file_paths)  # Call filter_data with the correct argument
        all_data.extend(data)  # Extend the all_data list with the results of filter_data

    # Splitting dataset into features and labels for training/testing
    
    

        for dir_name, file_name, points in data:
            features.append(get_features(points))
            labels.append(dir_name)
        
    for feature, label in zip(features, labels):
        print(f"Label: {label}, Features: {feature}")

    features = np.array(features)
    labels = np.array(labels)

    """Boiler plate code for splitting data into training and testing sets"""

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # print(f"Training set size: {len(X_train)}")
    # print(f"Testing set size: {len(X_test)}")

    # # sample 
    # print("Sample training data:", X_train[:5])
    # print("Sample testing data:", X_test[:5])


"""Sample code for Confusion Matrix to calculate precision, recall rate, and accuracy"""
# actual = numpy.random.binomial(1,.9,size = 1000)
# predicted = numpy.random.binomial(1,.9,size = 1000)

# confusion_matrix = metrics.confusion_matrix(actual, predicted)

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

# cm_display.plot()
# plt.show()


"""Sample code for KNN classifier"""

# n_neighbors = [5,15,25]
# accuracy_scores = []

# wine = datasets.load_wine()

# data = wine.data
# target = wine.target

# traindata = []
# traintarget = []

# testdata = []
# testtarget = []

# for i in range(0,29):
#     traindata.append(data[i])
#     traintarget.append(target[i])
# for i in range(29,59):
#     testdata.append(data[i])
#     testtarget.append(target[i])

# for i in range(59,89):
#     traindata.append(data[i])
#     traintarget.append(target[i])

# for i in range(89,119):
#     testdata.append(data[i])
#     testtarget.append(target[i])

# for i in range(119,144):
#     traindata.append(data[i])
#     traintarget.append(target[i])

# for i in range(144,178):
#     testdata.append(data[i])
#     testtarget.append(target[i])

# for n in n_neighbors:
#     knn = neighbors.KNeighborsClassifier(n)
#     knn.fit(traindata, traintarget)
#     predictions = knn.predict(testdata)
#     print(f"neighbors {n}", predictions)
#     accuracy = accuracy_score(testtarget, predictions)
#     print(f"Accuracy for neighbor {n}: {accuracy}")
#     accuracy_scores.append(accuracy)

# average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

# print("Average accuracy: ", average_accuracy)



if __name__ == '__main__':
    main()


