
import argparse
import numpy as np
from decision_tree_training import decision_tree_learning
import plot_decision_tree
import matplotlib.pyplot as plt

def main(filename):

    training_dataset = np.loadtxt(filename)
    root_node, max_depth = decision_tree_learning(training_dataset, 0)
    plot_decision_tree.plot_decision_tree(root_node)
    # save the plot to a file
    plt.savefig("decision_tree.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree Training")
    parser.add_argument("filename", type=str, help="Path to the training dataset file")
    args = parser.parse_args()
    
    main(args.filename)
    