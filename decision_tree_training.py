import argparse
import numpy as np
import matplotlib.pyplot as plt

import plot_decision_tree

class Node:
    def __init__(self, feature, value = 0, left=None, right=None, comparison=None):
        """
        Initialize a decision tree node.
        
        Args:
            value: The value stored in the node (either split value or class label)
            left: Left child node
            right: Right child node
            comparison: String describing the split condition (e.g., "x1 < 0.5")
        """
        self.left = left
        self.right = right
        self.feature = feature
        self.value = value
        self.comparison = comparison

    @staticmethod
    def get_max_depth_width(node, depth=0):
        """
        Calculate the maximum depth and width of the tree.
        
        Args:
            node: Root node of the tree/subtree
            depth: Current depth in the tree
            
        Returns:
            tuple: (width, depth) of the tree/subtree
        """
        if node is None:
            return 0, depth
        left_width, left_depth = Node.get_max_depth_width(node.left, depth + 1)
        right_width, right_depth = Node.get_max_depth_width(node.right, depth + 1)
        return left_width + right_width + 1, max(left_depth, right_depth)
    
    def __str__(self):
        """
        String representation of the node.
        
        Returns:
            str: Human-readable description of the node
        """
        if self.left is None and self.right is None:  # Leaf node
            return f"Leaf: {self.value}"
        else:  # Internal node
            comp_str = self.comparison if self.comparison else f"Split at {self.value}"
            left_str = str(self.left) if self.left else "None"
            right_str = str(self.right) if self.right else "None"
            return f"({comp_str}\n  L-> {left_str}\n  R-> {right_str})"

        """
        Calculates the entropy of a dataset 
        Entropy is calculated by summing the probabilities of each class label multiplied by the log probability of that same class label.
        
        Args: 
        - dataset: numpy array with labels
        
        Returns:
        - float: entropy of the dataset 
        """
def entropy(dataset):
    unique, counts = np.unique(dataset, return_counts=True)
    probabilities = counts / len(dataset)
    return -np.sum(probabilities * np.log2(probabilities))


    """Calculates the information gain gained by a split of the dataset into two subsets.

    Args:
    - s_all: numpy array with labels of the entire dataset
    - s_left: numpy array with labels of the left subset
    - s_right: numpy array with labels of the right subset
    
    Returns:
    - float: information gain of the split calculated using the entropy function
    """
def information_gain(s_all, s_left, s_right):
    total_size = s_left.size + s_right.size
    return entropy(s_all) - (s_left.size/total_size) * entropy(s_left) - (s_right.size/total_size) * entropy(s_right)
    
    
    """Find the best split point for a given attribute in the training dataset.
    
    Args: 
    - training_dataset: numpy array with features and labels
    - i: index of the attribute to find the split point
    
    Returns: 
    - (split_value, information_gain) : tuple, where split_value is the best value to split the dataset and information_gain is the information gain of the split.
    """
def find_split_point_attribute(training_dataset, i):
    attribute_values = training_dataset[:, i]
    label_values = training_dataset[:, -1]
    if len(np.unique(attribute_values)) == 1:
        return None, 0
    
    # Sort the attribute values so that it would be in increasing order
    # Example: [0,0,0,0,0,1,1,1,1,2....]
    sorted_indices = np.argsort(attribute_values)
    sorted_attribute_values = attribute_values[sorted_indices]
    sorted_label_values = label_values[sorted_indices]
    
    max_i = None
    max_information_gain = 0
    for j in range(1, len(sorted_attribute_values)):
        # If the label value changes, this is a potential split point
        # [0,0,0,0,0,1,1,1,1,2....] -> [0,0,0,0,0] [1,1,1,1,2....]
        if sorted_label_values[j] != sorted_label_values[j-1]:
            # Find the split value by averaging the two values and finding midpoint
            split_value = (sorted_attribute_values[j] + sorted_attribute_values[j-1]) / 2
            labels_left = sorted_label_values[:j]
            labels_right = sorted_label_values[j:]
            curr_gain = information_gain(sorted_label_values, labels_left, labels_right)
            if max_information_gain < curr_gain:
                max_information_gain = curr_gain
                max_i = split_value
    if max_i is None:
        return None, 0
    return max_i, max_information_gain

    """Find the best split across all attributes in the training dataset.
    Args:
    - training_dataset: numpy array with features and labels
    Returns:
    - (best_attribute, split_value): tuple, where best_attribute is the index of the best attribute to split the dataset, split_value is the best value to split the dataset
    """
def find_split(training_dataset):
    if len(training_dataset) < 2:
        raise ValueError("Dataset is too small")
    if training_dataset.shape[1] < 2:
        raise ValueError("Dataset has no attributes")
    best_attribute = None
    best_split_value = None
    max_information_gain = 0
    
    for i in range(training_dataset.shape[1] - 1):
        split_value, information_gain = find_split_point_attribute(training_dataset, i)
        if split_value is not None and information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute = i
            best_split_value = split_value
    if best_attribute is None:
        raise ValueError("No split found")
    return best_attribute, best_split_value

    """Recursively build a decision tree using the training dataset.
    
    Args:
    - training_dataset: numpy array with features and labels
    - depth: current depth of the tree, used for the recursion of the decision tree, initilaized with 0
    """
def decision_tree_learning(training_dataset, depth):
    # Check it is not a pure dataset, i.e only one class
    if np.all(training_dataset[:, -1] == training_dataset[:, -1][0]):
        return (Node(feature = training_dataset[:, -1][0]), depth + 1)

    split = find_split(training_dataset)
    l_dataset = training_dataset[training_dataset[:, split[0]] < split[1]]
    r_dataset = training_dataset[training_dataset[:, split[0]] >= split[1]]
    
    # If the split leads to a leaf, return the majority class
    if len(l_dataset) == 0 or len(r_dataset) == 0:
        # return the majority class and convert from float64 to int64
        counts = np.bincount(training_dataset[:, -1].astype(int))
        majority_class = np.argmax(counts)
        return (Node(feature = majority_class), depth + 1)
    
    
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    node = Node(feature = split[0], value = split[1], left = l_branch, right = r_branch, comparison= f"X[{split[0]}] < {split[1]}")
    return node, max(l_depth, r_depth)


