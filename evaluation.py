import argparse
import decision_tree_training
import numpy as np



class EvaluationMetrics:
    def __init__(self, dataset_filename, folds=10):
        self.matrix = self._cross_validation_confusion_matrix(dataset_filename, folds)
        self.folds = folds

    def _cross_validation_confusion_matrix(self, dataset_filename, folds):
        dataset = np.loadtxt(dataset_filename)
        np.random.shuffle(dataset)
        fold_size = len(dataset) // folds
        labels = np.unique(dataset[:, -1])
        confusion_matrices = np.empty((folds, len(labels), len(labels)))

        for i in range(folds):
            test_indices = list(range(i * fold_size, (i + 1) * fold_size))
            train_indices = list(set(range(len(dataset))) - set(test_indices))

            X_train = dataset[train_indices]
            X_test = dataset[test_indices]

            trained_tree, max_depth = decision_tree_training.decision_tree_learning(X_train, 0)
            predictions = self._predict(trained_tree, X_test)

            confusion_matrices[i] = self._generate_confusion_matrix(X_test[:, -1], predictions, labels)

        return np.mean(confusion_matrices, axis=0)

    @staticmethod
    def _generate_confusion_matrix(true_labels, predicted_labels, labels):
        confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            confusion_matrix[int(true_label) - 1, int(predicted_label) - 1] += 1

        return confusion_matrix

    @staticmethod
    def _predict(trained_tree, X_test):
        def evaluate_node(node, x):
            if node.left is None and node.right is None:
                return node.feature
            if x[node.feature] < node.value:
                return evaluate_node(node.left, x)
            else:
                return evaluate_node(node.right, x)

        predictions = np.array([evaluate_node(trained_tree, x) for x in X_test])
        return predictions

    def precision_rates(self):
        precision_rates = np.zeros(self.matrix.shape[0])
        for i in range(self.matrix.shape[0]):
            true_positives = self.matrix[i, i]
            false_positives = np.sum(self.matrix[:, i]) - true_positives
            if true_positives + false_positives == 0:
                precision_rates[i] = 0
            else:
                precision_rates[i] = true_positives / (true_positives + false_positives)
        return precision_rates

    def recall_rates(self):
        recall_rates = np.zeros(self.matrix.shape[0])
        for i in range(self.matrix.shape[0]):
            true_positives = self.matrix[i, i]
            false_negatives = np.sum(self.matrix[i, :]) - true_positives
            if true_positives + false_negatives == 0:
                recall_rates[i] = 0
            else:
                recall_rates[i] = true_positives / (true_positives + false_negatives)
        return recall_rates

    def f1_measures(self):
        """Calculates F1 measures for each label."""
        precision_rates = self.precision_rates()
        recall_rates = self.recall_rates()
        f1_measures = 2 * (precision_rates * recall_rates) / (precision_rates + recall_rates)
        f1_measures[np.isnan(f1_measures)] = 0  # Handling division by zero cases
        return f1_measures

    @staticmethod
    def macro_average(rates):
        """Calculates macro-averaged rates."""
        return np.mean(rates)

    def print_summary(self):
        """Prints a summary of the metrics."""
        print("Confusion Matrix:\n", self.matrix)
        print("Precision rates:", self.precision_rates())
        print("Macro-averaged precision:", self.macro_average(self.precision_rates()))
        print("Recall rates:", self.recall_rates())
        print("Macro-averaged recall:", self.macro_average(self.recall_rates()))
        print("F1 measures:", self.f1_measures())
        print("Macro-averaged F1 score:", self.macro_average(self.f1_measures()))
        print("Number of rows for each class:: ", np.sum(self.matrix, axis=1))




"""Evaluation of the model Function .
        Parameters:
        -----------
        trained_tree: RootNode - The root node of the trained decision tree
        X_test: np.ndarray - The test dataset
        
        Returns:
        Accuracy: float - The accuracy of the model
"""
def evaluate(trained_tree, X_test):
    def evaluate_node(node, x):
        if node.left is None and node.right is None:
            return node.feature
        if x[node.feature] < node.value:
            return evaluate_node(node.left, x)
        else:
            return evaluate_node(node.right, x)

    predictions = np.array([evaluate_node(trained_tree, x) for x in X_test])
    return np.mean(predictions == X_test[:, -1])


"""Cross Validation Function  using 10-folds.
        Parameters:
"""


def cross_validation_evaluation(dataset_filename, folds=10):
    dataset = np.loadtxt(dataset_filename)
    np.random.shuffle(dataset)
    fold_size = len(dataset) // folds
    accuracies = []

    for i in range(folds):
        test_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = list(set(range(len(dataset))) - set(test_indices))

        X_train = dataset[train_indices]
        X_test = dataset[test_indices]

        trained_tree, max_depth = decision_tree_training.decision_tree_learning(X_train, 0)
        accuracies.append(evaluate(trained_tree, X_test))

    return np.mean(accuracies)


def main_two_arguments(filename1 = "clean_dataset.txt", filename2 = "noisy_dataset.txt"):
    print("Computing Evaluations of both datasets")
    print("Accuracies dataset 1: ", cross_validation_evaluation(filename1))
    print("Accuracies dataset 2: ", cross_validation_evaluation(filename2))
    print("Beginning computation")

    print(f"Dataset {filename1}") 
    clean_metrics = EvaluationMetrics(filename1)
    clean_metrics.print_summary()

    print(f"Dataset {filename2}") 
    noisy_metrics = EvaluationMetrics(filename2)
    noisy_metrics.print_summary()
    
def main_one_argument(filename = "clean_dataset.txt"):
    print("Computing Evaluations of one datasets")
    print("Accuracy: ", cross_validation_evaluation(filename))
    print("Beginning computation")

    print(f"Dataset: {filename}")
    clean_metrics = EvaluationMetrics(filename)
    clean_metrics.print_summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree Evaluation")
    parser.add_argument("filename", type=str, help="Path to the first training dataset file")
    # Not required second argument
    parser.add_argument("filename2", type=str, help="Path to the second training dataset file", nargs='?')
    sys = parser.parse_args()
    if sys.filename2:
        main_two_arguments(sys.filename, sys.filename2)
    else:
        main_one_argument(sys.filename)