#!/usr/bin/env python3

"""
Face Finder for Face Recognition

This script contains the ModelTrainer class which is used to train and evaluate
different machine learning models for face recognition.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from data_processor import DataProcessor
from report_manager import ReportManager
from sklearn.preprocessing import LabelBinarizer


class ModelTrainer(DataProcessor, ReportManager):
    def __init__(self):
        DataProcessor.__init__(self)
        ReportManager.__init__(self)

    def naives_bayes(self, training_features: List[float], training_labels: List[int],
                     testing_features: List[float], testing_labels: List[int]
                     ) -> None:
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
        report = classification_report(
            testing_labels, predictions, zero_division=0)
        return report, testing_labels, predictions, 'Naives Bayes'

    def ann(self, training_features: List[float], training_labels: List[int],
            testing_features: List[float], testing_labels: List[int]
            ) -> None:

        classifier = MLPClassifier(hidden_layer_sizes=(
            100,), max_iter=10000, activation='relu', solver='adam', random_state=1)
        classifier.fit(training_features, training_labels)
        predictions = classifier.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)

        return report, testing_labels,  predictions, 'ANN'

    def knn(self, training_features: List[float], training_labels: List[int],
            testing_features: List[float], testing_labels: List[int]
            ) -> tuple[List[int], List[int], str]:
        """
        Trains and evaluates a k-Nearest Neighbors (kNN) classifier.

        Args:
            training_features (List): Training feature vectors.
            training_labels (List): Training labels.
            testing_features (List): Testing feature vectors.
            testing_labels (List): Testing labels.
        """

        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(training_features, training_labels)
        predictions = knn.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)

        return report, testing_labels, predictions, 'KNN'

    def support_vector_machine(self, training_features, training_labels, testing_features, testing_labels):
        classifier = SVC(kernel='rbf', C=1)
        classifier.fit(training_features, training_labels)
        predictions = classifier.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)
        return report, testing_labels, predictions, 'SVM'

    def decision_tree(self, training_features, training_labels, testing_features, testing_labels):
        tree_classifier = DecisionTreeClassifier(random_state=0)
        tree_classifier.fit(training_features, training_labels)
        predictions = tree_classifier.predict(testing_features)
        report = classification_report(
            testing_labels, predictions, zero_division=0)
        return report, testing_labels, predictions, 'Decision Tree'

    def evaluation_metrics(self, testing_labels, predictions,
                           model_name, output_dir='confusion_matrices_plt') -> tuple[float, np.ndarray, str]:
        """
        Plots a confusion matrix for a multiclass classification problem.

        Args:
            testing_labels (List): True labels of the testing set.
            predictions (List): Predicted labels for the testing set.
            model_name (str): Name of the model for which the confusion matrix is being plotted.
            output_dir (str): Directory to save the confusion matrix plot. Default is 'confusion_matrices'.

        Returns:
            tuple: A tuple containing the accuracy score, confusion matrix, and model name.
        """

        accuracy = accuracy_score(
            testing_labels, predictions)
        conf_matrix = confusion_matrix(testing_labels, predictions)

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_name}')
        file_path = os.path.join(output_dir, f'CM_{model_name}.png')
        plt.savefig(file_path)
        plt.pause(.1)
        plt.close()

        return accuracy, conf_matrix, model_name

    def roc_curve(self, testing_labels, predictions,
                  model_name, output_dir='roc_curves_plt'
                  ) -> tuple[dict, dict, dict]:
        """
        Plots One-vs-Rest (OvR) ROC curves for a multiclass classification problem.

        Args:
            testing_labels (List): True labels of the testing set.
            predictions (List): Predicted probabilities or decision function scores for the testing set.
            model_name (str): Name of the model for which the ROC curves are being plotted.
            output_dir (str): Directory to save the ROC curve plot. Default is 'roc_curves'.

        Returns:
            tuple: A tuple containing the false positive rate, true positive rate, and ROC area.
        """
        os.makedirs(output_dir, exist_ok=True)

        lb = LabelBinarizer()
        lb.fit(testing_labels)
        y_true = lb.transform(testing_labels)

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(lb.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], predictions[:, 0])
            roc_auc[i] = auc(fpr[i], tpr[i])

        #  plot the roc curve for the model and save it to a file in the output directory
        plt.figure()
        colors = ['blue', 'red', 'green', 'purple',
                  'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']
        for i, color in zip(range(len(lb.classes_)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve (class {lb.classes_[i]}) (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f'ROC_{model_name}.png'))
        plt.close()
        return fpr, tpr, roc_auc
