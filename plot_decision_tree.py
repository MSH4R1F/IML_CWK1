import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_decision_tree(root_node, figsize=(15, 10)):
    def get_tree_layout(node, depth=0, pos=0, min_sep=2.0, positions=None, depths=None):
        """
        Recursively calculate the layout of the tree, including positions and depths of nodes.
        """
        if positions is None:
            positions = {}
        if depths is None:
            depths = {}
            
        if node is None:
            return positions, depths, 0
            
        depths[node] = depth
        
        left_width = 0
        right_width = 0
        
        if node.left:
            positions_l, depths, left_width = get_tree_layout(
                node.left, depth + 1, pos, min_sep, positions, depths)
        if node.right:
            positions_r, depths, right_width = get_tree_layout(
                node.right, depth + 1, pos + left_width + min_sep, min_sep, positions, depths)
            
        width = max(min_sep, left_width + right_width)
        positions[node] = pos + width / 2
        
        return positions, depths, width

    def plot_node(node, positions, depths, ax):
        """
        Recursively plot the tree nodes and edges on the given matplotlib axes.
        """
        if node is None:
            return
            
        x = positions[node]
        y = 1 - depths[node] * 0.15
        
        if node.left is None and node.right is None:
            node_text = f"leaf:{node.value:.3f}"
        else:
            node_text = node.comparison
            
        box_width = max(0.1, len(node_text) * 0.5)  # Ensures minimum width
        box_height = 0.06
        
        box = plt.Rectangle((x - box_width/2, y - box_height/2), 
                          box_width, box_height, 
                          facecolor='white',
                          edgecolor='blue',
                          linewidth=1,
                          zorder=2)
        ax.add_patch(box)
        
        ax.text(x, y, node_text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8, 
                zorder=3)
        
        if node.left:
            child_x = positions[node.left]
            child_y = 1 - depths[node.left] * 0.15
            ax.plot([x, child_x], [y - box_height/2, child_y + box_height/2],
                   'k-', linewidth=1, zorder=1)
            plot_node(node.left, positions, depths, ax)
            
        if node.right:
            child_x = positions[node.right]
            child_y = 1 - depths[node.right] * 0.15
            ax.plot([x, child_x], [y - box_height/2, child_y + box_height/2],
                   'k-', linewidth=1, zorder=1)
            plot_node(node.right, positions, depths, ax)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    positions, depths, _ = get_tree_layout(root_node)
    plot_node(root_node, positions, depths, ax)

