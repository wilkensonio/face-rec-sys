import os
import numpy as np
import os
import numpy as np
from plot import Plot


"""
Report Manager for Face Recognition
This class is responsible for generating and writing the 
classification report for the Face Recognition project.
And also for plotting the confusion matrix.
"""


class ReportManager(Plot):
    def __init__(self):
        super().__init__()

    def classification_report(self, report, accuracy, conf_matrix, model_name, type='',
                              output_dir='classification_reports'
                              ) -> None:
        report_file = os.path.join(output_dir, f'report_{model_name}.txt')
        with open(report_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
            f.write(f"Classification Report:\n{report}\n\n")
            f.write(f"{model_name} Confusion Matrix:\n")
            np.savetxt(f, conf_matrix, fmt='%d')
            f.write("\n")

    def roc_report(self, fpr, tpr, roc_auc, model_name,
                   output_dir='roc_repot') -> None:
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, f'roc_report_{model_name}.txt')
        with open(report_file, 'w') as f:
            f.write(f"Model: {model_name}\n\n")
            for i, cls in enumerate(fpr):
                f.write(f"Class {i} ROC Curve (area = {roc_auc[i]:.2f}):\n")
                f.write(f"False Positive Rate: {fpr[i]}\n")
                f.write(f"True Positive Rate : {tpr[i]}\n\n")

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

        fpr_ann, tpr_ann, roc_auc_ann = self.plt_roc_curve(testing_labels_ann,
                                                           predictions_ann, model_name_ann)

        self.classification_report(report_ann, accuracy_ann,
                                   conf_matrix_ann, model_name_ann)

        self.roc_report(fpr_ann, tpr_ann, roc_auc_ann, model_name_ann)

        # KNN model
        report_knn, testing_labels_knn, predictions_knn, model_name_knn = self.knn(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_knn, conf_matrix_knn, model_name_knn = self.evaluation_metrics(
            testing_labels_knn, predictions_knn, model_name_knn)

        fpr_knn, tpr_knn, roc_auc_knn = self.plt_roc_curve(testing_labels_knn,
                                                           predictions_knn, model_name_knn)

        self.classification_report(report_knn, accuracy_knn,
                                   conf_matrix_knn, model_name_knn)

        self.roc_report(fpr_knn, tpr_knn, roc_auc_knn, model_name_knn)

        # Naive Bayes model
        report_nb, testing_labels_nb, predictions_nb, model_name_nb = self.naives_bayes(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_nb, conf_matrix_nb, model_name_nb = self.evaluation_metrics(
            testing_labels_nb, predictions_nb, model_name_nb)

        fpr_nb, tpr_nb, roc_auc_nb = self.plt_roc_curve(
            testing_labels_nb, predictions_nb, model_name_nb)

        self.classification_report(report_nb, accuracy_nb,
                                   conf_matrix_nb, model_name_nb)

        self.roc_report(fpr_nb, tpr_nb, roc_auc_nb, model_name_nb)

        # SVM model
        report_svm, testing_labels_svm, predictions_svm, model_name_svm = self.support_vector_machine(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_svm, conf_matrix_svm, model_name_svm = self.evaluation_metrics(
            testing_labels_svm, predictions_svm, model_name_svm)

        fpr_svm, tpr_svm, roc_auc_svm = self.plt_roc_curve(testing_labels_svm,
                                                           predictions_svm, model_name_svm)

        self.classification_report(report_svm, accuracy_svm,
                                   conf_matrix_svm, model_name_svm)

        self.roc_report(fpr_svm, tpr_svm, roc_auc_svm, model_name_svm)

        # Decision Tree model
        report_dt, testing_labels_dt, predictions_dt, model_name_dt = self.decision_tree(
            training_features, training_labels, testing_features, testing_labels)

        accuracy_dt, conf_matrix_dt, model_name_dt = self.evaluation_metrics(
            testing_labels_dt, predictions_dt, model_name_dt)

        fpr_dt, tpr_dt, roc_auc_dt = self.plt_roc_curve(
            testing_labels_dt, predictions_dt, model_name_dt)

        self.classification_report(report_dt, accuracy_dt,
                                   conf_matrix_dt, model_name_dt)

        self.roc_report(fpr_dt, tpr_dt, roc_auc_dt, model_name_dt)
        Plot(training_features, training_labels, testing_features,
             testing_labels).plot_classifier()

        print("✅ Report generated successfully 🗂️ 🗂️\n")
        print("1. Check the classification_reports folder for the report\n")
        print("2. Check the roc_curve_plt folder for the ROC curve plots\n")
        print(
            "3. Check the confusion_matrices_plt folder for the confusion matrices plots\n")
        print("The folder can be found in the root directory of the project\n")
