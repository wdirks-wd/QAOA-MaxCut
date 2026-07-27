import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Generates unweighted graph with variable edge probability.
def generate_unweighted_graph(num_nodes, edge_probability, seed=None):
    rng = np.random.default_rng(seed)
    w = np.zeros((num_nodes, num_nodes), dtype=float)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() <= edge_probability:
                weight = 1.0
                w[i, j] = weight
                w[j, i] = weight
    return w

# Generates unweighted N-Regular graph.
def generate_unweighted_random_regular_graph(num_nodes, graph_seed, degree):
    G = nx.random_regular_graph(d=degree, n=num_nodes, seed=graph_seed)
    sparse_matrix = nx.adjacency_matrix(G)
    w = sparse_matrix.toarray()
    return w

# Perturbs the weights of the edges of the input graph adjacency matrix. The weights remain unsigned, W >= 0.
def perturb_weight_unsigned(weight_matrix, location, scale, perturbation_seed):
    rng = np.random.default_rng(perturbation_seed)
    num_nodes = weight_matrix.shape[0]
    W = weight_matrix.astype(float)
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if weight_matrix[i,j] != 0:
                    
                delta = rng.normal(loc=location, scale=scale)
                W[i,j] = abs(weight_matrix[i,j] + delta)
                W[j,i] = W[i,j]
    return W

# Perturbs the weights of the edges of the input graph adjacency matrix. The weights are capable of taking on any real value.
def perturb_weight_signed(weight_matrix, location, scale, perturbation_seed):
    rng = np.random.default_rng(perturbation_seed)
    num_nodes = weight_matrix.shape[0]
    W = weight_matrix.astype(float)
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if weight_matrix[i,j] != 0:
                    
                delta = rng.normal(loc=location, scale=scale)
                W[i,j] = weight_matrix[i,j] + delta
                W[j,i] = W[i,j]
    return W