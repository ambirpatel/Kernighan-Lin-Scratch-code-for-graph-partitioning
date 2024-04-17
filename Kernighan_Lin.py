import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
# from matplotlib import pyplot as plt

# Top-level overview:
# This script uses the pandas, numpy, networkx, and tqdm libraries to implement the Kernighan-Lin graph partitioning algorithm on a graph represented by an adjacency matrix.
# The goal is to partition the graph's nodes into two disjoint sets such that the cut size (number of edges crossing between the sets) is minimized.

# Read edge data from file
data = pd.read_csv("karate.dat", sep=' ')

# Number of vertices
n = max(list(data.max())) 

# Adjacency matrix
A = np.zeros((n + 1, n + 1))

# Create adjacency matrix from the data
for i in range(data.shape[0]):
    x = data.iloc[i, 0]
    y = data.iloc[i, 1]
    A[x, y] = 1
    A[y, x] = 1
    
# List of vertices
v = list(range(n))

# Divide nodes into two random groups
random.shuffle(v)
cut = int(n / 2)
n1, n2 = np.sort(v[:cut]), np.sort(v[cut:])
pairs = len(n1) * len(n2)

def cut_size(n1, n2):
    """Calculate the cut size (number of edges between nodes in two groups).

    Parameters:
    n1 (list): First group of nodes.
    n2 (list): Second group of nodes.

    Returns:
    int: The cut size (number of edges between nodes in groups n1 and n2).
    """
    r_old = 0
    for i in n1:
        for j in n2:
            if A[i, j] == 1:
                r_old += 1
    return r_old

def best_swap(n1, n2):
    """Find the best swap between groups to minimize cut size.

    Parameters:
    n1 (list): First group of nodes.
    n2 (list): Second group of nodes.

    Returns:
    tuple: Minimum cut size and the corresponding partition.
    """
    mark = []
    partition = []
    cut_s = []
    while len(mark) != pairs - 1:
        r_best, ij, part = [], [], []
        r_old = cut_size(n1, n2)
        for i in range(len(n1)):
            for j in range(len(n2)):
                if ([n1[i], n2[j]] not in mark and [n2[j], n1[i]] not in mark):
                    n11, n22 = n1.copy(), n2.copy()
                    n11[i] = n2[j]
                    n22[j] = n1[i]
                    del_R = r_old - cut_size(n11, n22)  # Change in cut size
                    r_best.append(del_R)
                    part.append([n11, n22])
                    ij.append([n11[i], n22[j]])
        
        a = np.argmax(r_best)
        c_part = part[a]
        partition.append(c_part)
        mark.append(ij[a])
        cut_s.append(cut_size(c_part[0], c_part[1]))
        n1, n2 = c_part[0], c_part[1]
    return cut_s[np.argmin(cut_s)], partition[np.argmin(cut_s)]

def kernighan_lin(n1, n2):
    """Perform Kernighan-Lin partitioning to minimize cut size.

    Parameters:
    n1 (list): First group of nodes.
    n2 (list): Second group of nodes.

    Returns:
    tuple: Final cut size and partition.
    """
    cs = []
    for _ in range(1):  # Number of iterations
        CutSize, partition = best_swap(n1, n2)
        n1, n2 = partition[0], partition[1]
        cs.append(CutSize)
        # Check if cut size is converging
        if len(cs) >= 10:
            ts = cs[-5:]
            if ts[0] == ts[1] == ts[2] == ts[3] == ts[4]:
                break
    return CutSize, partition

# Perform Kernighan-Lin partitioning
c, p = kernighan_lin(n1, n2)
p1, p2 = np.sort(p[0]), np.sort(p[1])
print(f"\nCut size is : {c}")
print(f"Partition 1 : {p1}")
print(f"Partition 2 : {p2}")

# Define groups
grp1 = p[0]
grp2 = p[1]

# Create a NetworkX graph
G = nx.Graph()

# Add edges to the graph
for i in range(len(data)):
    G.add_edge(data.iloc[i, 0], data.iloc[i, 1])

# Color nodes based on partition
for n in G.nodes():
    G.nodes[n]['color'] = 'b' if n in grp1 else 'g'

# Create color list for nodes
colors = [node[1]['color'] for node in G.nodes(data=True)]

# Draw the graph with labels and colors
nx.draw(G, with_labels=True, node_color=colors)
