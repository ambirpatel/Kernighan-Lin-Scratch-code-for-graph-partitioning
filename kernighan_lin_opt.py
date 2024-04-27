import pandas as pd
import numpy as np
import random
import networkx as nx

# Read edge data from file
# data
data = pd.read_csv("karate.dat", sep=' ')

# Number of vertices
n = max(data.max())

# Create adjacency list from the data
adj_list = {i: set() for i in range(1, n + 1)}
for _, row in data.iterrows():
    x, y = row[0], row[1]
    adj_list[x].add(y)
    adj_list[y].add(x)

# Divide nodes into two random groups
nodes = list(range(1, n + 1))
random.shuffle(nodes)
cut = n // 2
n1 = set(nodes[:cut])
n2 = set(nodes[cut:])

def cut_size(n1, n2):
    """Calculate the cut size (number of edges between nodes in two groups).

    Parameters:
    n1 (set): First group of nodes.
    n2 (set): Second group of nodes.

    Returns:
    int: The cut size (number of edges between nodes in groups n1 and n2).
    """
    # Calculate cut size using adjacency list
    cut_size = sum(1 for node in n1 for neighbor in adj_list[node] if neighbor in n2)
    return cut_size

def best_swap(n1, n2):
    """Find the best swap between groups to minimize cut size.

    Parameters:
    n1 (set): First group of nodes.
    n2 (set): Second group of nodes.

    Returns:
    tuple: Minimum cut size and the corresponding partition.
    """
    best_cut_size = cut_size(n1, n2)
    best_partition = (n1, n2)
    best_change = 0

    # Check all possible swaps
    for node1 in n1:
        for node2 in n2:
            # Swap the nodes
            n1.remove(node1)
            n1.add(node2)
            n2.remove(node2)
            n2.add(node1)

            # Calculate new cut size
            new_cut_size = cut_size(n1, n2)
            change = best_cut_size - new_cut_size

            # Check if this swap is beneficial
            if change > best_change:
                best_cut_size = new_cut_size
                best_change = change
                best_partition = (n1.copy(), n2.copy())

            # Revert the swap
            n1.remove(node2)
            n1.add(node1)
            n2.remove(node1)
            n2.add(node2)

    return best_cut_size, best_partition

def kernighan_lin(n1, n2):
    """Perform Kernighan-Lin partitioning to minimize cut size.

    Parameters:
    n1 (set): First group of nodes.
    n2 (set): Second group of nodes.

    Returns:
    tuple: Final cut size and partition.
    """
    for _ in range(10):  # Number of iterations or until convergence
        cut_size, partition = best_swap(n1, n2)
        n1, n2 = partition
    return cut_size, partition

# Perform Kernighan-Lin partitioning
cut_size, partition = kernighan_lin(n1, n2)
p1, p2 = sorted(partition[0]), sorted(partition[1])
print(f"Cut size: {cut_size}")
print(f"Partition 1: {p1}")
print(f"Partition 2: {p2}")

# Create a NetworkX graph
G = nx.Graph()
G.add_edges_from(data.values)

# Define colors based on partitions
colors = ['b' if node in p1 else 'g' for node in G.nodes()]

# Draw the graph with labels and colors
nx.draw(G, with_labels=True, node_color=colors)
