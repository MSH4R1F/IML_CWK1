import numpy as np
import matplotlib.pyplot as plt

class LeafNode:
    def __init__(self, value):
        self.value = value

class Node:
    def __init__(self,value, left, right):
        self.left = left
        self.right = right
        self.value = value

"""

"""
def find_split_points(feature_values):
    sorted_values = sorted(feature_values)

    pass

"""
l
"""
def find_split(training_dataset):
    pass

def decision_tree_learning(training_dataset, depth):
    # check it is not a pure dataset
    if np.all(training_dataset[:, -1] == training_dataset[:, -1][0]):
        return (LeafNode(training_dataset[:, -1][0]), depth + 1)

    split = find_split(training_dataset)
    l_branch = todo
    r_branch = todo
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    node = (Node(split, l_branch, r_branch), max(l_depth, r_depth))