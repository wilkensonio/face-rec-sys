import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

"""
Report Manager for Face Recognition
This class is responsible for generating and writing the 
classification report for the Face Recognition project.
And also for plotting the confusion matrix.
"""


class ReportManager:
    def write_report(self, report, accuracy, conf_matrix, model_name, output_dir='classification_reports'):
        report_file = os.path.join(output_dir, f'report_{model_name}.txt')
        with open(report_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
            f.write(f"Classification Report:\n{report}\n\n")
            f.write(f"{model_name} Confusion Matrix:\n")
            np.savetxt(f, conf_matrix, fmt='%d')
            f.write("\n")

    def report(self, file_dict: dict[str, str], encoded_labels: list[int],
               output_dir='classification_reports') -> None:

        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        training_features, training_labels, testing_features, testing_labels = self.organize(
            self.training_features, self.training_labels,
            self.testing_features, self.testing_labels, file_dict, encoded_labels)

        print("\nReport is being generated ...\n")
        # ANN model
        report_ann, testing_labels_ann, predictions_ann, model_name_ann = self.ann(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_ann, conf_matrix_ann, model_name_ann = self.evaluation_metrics(
            testing_labels_ann, predictions_ann, model_name_ann)

        self.write_report(report_ann, accuracy_ann,
                          conf_matrix_ann, model_name_ann)

        # KNN model
        report_knn, testing_labels_knn, predictions_knn, model_name_knn = self.knn(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_knn, conf_matrix_knn, model_name_knn = self.evaluation_metrics(
            testing_labels_knn, predictions_knn, model_name_knn)

        self.write_report(report_knn, accuracy_knn,
                          conf_matrix_knn, model_name_knn)

        # Naive Bayes model
        report_nb, testing_labels_nb, predictions_nb, model_name_nb = self.naives_bayes(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_nb, conf_matrix_nb, model_name_nb = self.evaluation_metrics(
            testing_labels_nb, predictions_nb, model_name_nb)

        self.write_report(report_nb, accuracy_nb,
                          conf_matrix_nb, model_name_nb)

        # SVM model
        report_svm, testing_labels_svm, predictions_svm, model_name_svm = self.support_vector_machine(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_svm, conf_matrix_svm, model_name_svm = self.evaluation_metrics(
            testing_labels_svm, predictions_svm, model_name_svm)

        self.write_report(report_svm, accuracy_svm,
                          conf_matrix_svm, model_name_svm)

        # Decision Tree model
        report_dt, testing_labels_dt, predictions_dt, model_name_dt = self.decision_tree(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_dt, conf_matrix_dt, model_name_dt = self.evaluation_metrics(
            testing_labels_dt, predictions_dt, model_name_dt)

        self.write_report(report_dt, accuracy_dt,
                          conf_matrix_dt, model_name_dt)

        print("Report generated successfully 🗂️ 🗂️\n")
        print("Check the classification_reports folder for the report")
        print("Check the confusion_matrices_plt folder for the confusion matrices plots\n")
