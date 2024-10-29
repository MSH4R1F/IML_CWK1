import decision_tree
import numpy as np
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
def cross_validation_evaluation(dataset_filename, folds = 10):
    dataset = np.loadtxt(dataset_filename)
    np.random.shuffle(dataset)
    fold_size = len(dataset) // folds 
    accuracies = []
    
    for i in range(folds):
        test_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = list(set(range(len(dataset))) - set(test_indices))
        
        X_train = dataset[train_indices]
        X_test = dataset[test_indices]
        
        trained_tree, max_depth = decision_tree.decision_tree_learning(X_train, 0)
        accuracies.append(evaluate(trained_tree, X_test))
        
    return np.mean(accuracies)

"""Prediction Function, returns the predicted labels of the test dataset."""
def predict(trained_tree, X_test):
    def evaluate_node(node, x):
        if node.left is None and node.right is None:
            return node.feature
        if x[node.feature] < node.value:
            return evaluate_node(node.left, x)
        else:
            return evaluate_node(node.right, x)
        
    predictions = np.array([evaluate_node(trained_tree, x) for x in X_test])
    return predictions

"""Cross Validation Function  using 10-folds. Returns the average confusion matrix of the model."""
def cross_validation_confusion_matrix(dataset_filename, folds = 10) : 
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
        
        
        trained_tree, max_depth = decision_tree.decision_tree_learning(X_train, 0)
        predictions = predict(trained_tree, X_test)
        
        confusion_matrices[i] = generate_confusion_matrix(X_test[:, -1], predictions, labels)
        
    return np.mean(confusion_matrices, axis=0)


def generate_confusion_matrix(true_labels, predicted_labels, labels):
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[int(true_label) - 1, int(predicted_label) - 1] += 1
    
    return confusion_matrix


def calculate_precision_rates(confusion_matrix):
    precision_rates = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        if true_positives + false_positives == 0:
            precision_rates[i] = 0
        else:
            precision_rates[i] = true_positives / (true_positives + false_positives)
    return precision_rates

def calculate_recall_rates(confusion_matrix):
    recall_rates = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[i, i]
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
        if true_positives + false_negatives == 0:
            recall_rates[i] = 0
        else:
            recall_rates[i] = true_positives / (true_positives + false_negatives)
    return recall_rates

def calculate_macro_average(rates):
    return np.mean(rates)

def calculate_f1_measures(precision_rates, recall_rates):
    f1_measures = 2 * (precision_rates * recall_rates) / (precision_rates + recall_rates)
    return f1_measures

if __name__ == "__main__":
    print("Accuracies: ", cross_validation_evaluation("clean_dataset.txt"))
    print("Accuracies noisy set: ", cross_validation_evaluation("noisy_dataset.txt"))
    print("Beginning computation")
    clean_dataset_confusion_matrix = cross_validation_confusion_matrix("clean_dataset.txt")
    noisy_dataset_confusion_matrix = cross_validation_confusion_matrix("noisy_dataset.txt")
    print("Confusion matrix clean dataset: ", clean_dataset_confusion_matrix)
    print("Confusion matrix noisy dataset: ", noisy_dataset_confusion_matrix)
    clean_dataset_precision_rates = calculate_precision_rates(clean_dataset_confusion_matrix)
    noisy_dataset_precision_rates = calculate_precision_rates(noisy_dataset_confusion_matrix)
    clean_dataset_recall_rates = calculate_recall_rates(clean_dataset_confusion_matrix)
    noisy_dataset_recall_rates = calculate_recall_rates(noisy_dataset_confusion_matrix)
    print("Precision rates Clean dataset: ",clean_dataset_precision_rates)
    print("MacroAveraged clean dataset: ", calculate_macro_average(clean_dataset_precision_rates))
    print("Precision rates noisy set: ", calculate_precision_rates(noisy_dataset_confusion_matrix))
    print("MacroAveraged noisy dataset: ", calculate_macro_average(noisy_dataset_precision_rates))
    
    # recall rates
    print("Recall rates Clean dataset: ", clean_dataset_recall_rates)
    print("MacroAveraged clean dataset: ", calculate_macro_average(clean_dataset_recall_rates))
    print("Recall rates noisy set: ", calculate_recall_rates(noisy_dataset_confusion_matrix))
    print("MacroAveraged noisy dataset: ", calculate_macro_average(noisy_dataset_recall_rates))
    
    #f1measures
    f1_measures_clean = 2 * (clean_dataset_precision_rates * clean_dataset_recall_rates) / (clean_dataset_precision_rates + clean_dataset_recall_rates)
    f1_measures_noisy = 2 * (noisy_dataset_precision_rates * noisy_dataset_recall_rates) / (noisy_dataset_precision_rates + noisy_dataset_recall_rates)
    print("F1 measures clean dataset: ", f1_measures_clean)
    print("MacroAveraged clean dataset: ", calculate_macro_average(f1_measures_clean))
    print("F1 measures noisy dataset: ", f1_measures_noisy)
    print("MacroAveraged noisy dataset: ", calculate_macro_average(f1_measures_noisy))
    
    ## Number of rows for each clas
    
    # Clean dataset
    print("Number of rows for each class in clean dataset: ", np.sum(clean_dataset_confusion_matrix, axis=1))
    # Noisy dataset
    print("Number of rows for each class in noisy dataset: ", np.sum(noisy_dataset_confusion_matrix, axis=1))
    