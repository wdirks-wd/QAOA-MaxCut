import numpy as np
import networkx as nx

# Returns binary string for n with length num_nodes.
def bit(n, num_bits):
    return np.binary_repr(n, num_bits)

# Returns binary string for n with length num_nodes in vector format.
def bitfield(n, num_bits):
    return np.array([int(bit) for bit in np.binary_repr(n, num_bits)],dtype=int)

# Calculates cut value for a given bitstring.
def cut_size_from_graph(graph, bitstring):
    assignment = [node for node, bit in zip(graph.nodes(), bitstring[::-1]) if bit == "1"]
    return nx.cut_size(graph, assignment, weight="weight")

# Calculates the weighted objective function for an input bitstring X.
def weighted_objective_value(x, w):
    # X[i, j] is 1 if node i and node j are in different partitions, 0 if nodes are in same partition.
    X = np.abs(np.subtract.outer(x, x))
    # Piecewise multiply weights and bitstring array, then sum.
    return np.sum(w * X) / 2

# Brute force approach to solving max cut by calculating the edge cut value of every partition.
def classical_max_cut(weight_matrix):
    num_nodes = weight_matrix.shape[0]
    maximum = 2**num_nodes
    sol = -np.inf
    best_cut = None

    for i in range(maximum):
        cur = bitfield(i, num_nodes)
        cur_v = weighted_objective_value(cur, weight_matrix)
        if cur_v > sol:
            sol = cur_v
            best_cut = cur
    return best_cut, sol