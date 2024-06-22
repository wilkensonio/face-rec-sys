#!/usr/bin/env python3

"""
Main script for running the Face Recognition project 
@author: Wilkenson | Kyle | Aeron
@date: 01/01/2024
@version: 1.0
README.md file for more information
"""

from model_trainer import ModelTrainer
from sklearn.preprocessing import LabelEncoder


def main():
    face_finder = ModelTrainer()
    file_dict = face_finder.all_file_paths()

    # List of labels in text form before we encode them
    all_labels = list(file_dict.keys())

    # Encode the labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Create a dictionary of encoded labels
    encoded_labels = {dir_name: label_encoder.transform(
        [dir_name])[0] for dir_name in file_dict.keys()}

    # Generate the report sand write it to a file
    face_finder.report(file_dict, encoded_labels)


if __name__ == '__main__':
    main()
