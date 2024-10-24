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

if __name__ == "__main__":
    print("Accuracies: ", cross_validation_evaluation("clean_dataset.txt"))
    print("Accuracies noisy set: ", cross_validation_evaluation("noisy_dataset.txt"))