import numpy as np
import graph as g
import classical as c
import hamiltonian as h
import qaoa_solver as q

# Runs a one-shot QAOA trial for an unweighted graph.
def run_unweighted_trial(num_nodes, 
              reps,
              graph_seed, 
              degree,
              sampler_seed, 
              optimizer_seed
              ):
    
    adjacency_matrix = g.generate_unweighted_random_regular_graph(num_nodes, graph_seed, degree)
    optimal_bitstring, optimal_value = c.classical_max_cut(weight_matrix=adjacency_matrix)
    cost_hamiltonian = h.build_maxcut_hamiltonian(weight_matrix=adjacency_matrix)
    qaoa = q.QAOA_Build(repetitions=reps, sampler_seed=sampler_seed, optimizer_seed=optimizer_seed)
    most_likely_eigenstate, QAOA_Result = q.QAOA_Output(cost_hamiltonian, qaoa)
    expectation_value, ratio = q.approximation_ratio(most_likely_eigenstate, optimal_value, adjacency_matrix)
    return adjacency_matrix, optimal_bitstring, optimal_value, most_likely_eigenstate, ratio, expectation_value

# Runs a one-shot QAOA trial for a weighted graph.
def run_weighted_trial(num_nodes, 
              reps,
              sigma,
              perturbation_seed,
              graph_seed, 
              degree,
              sampler_seed, 
              optimizer_seed
              ):

    adjacency_matrix = g.generate_unweighted_random_regular_graph(num_nodes, graph_seed, degree)
    weight_matrix = g.perturb_weight_unsigned(weight_matrix=adjacency_matrix, location=0, scale=sigma, perturbation_seed=perturbation_seed)
    optimal_bitstring, optimal_value = c.classical_max_cut(weight_matrix=weight_matrix)
    cost_hamiltonian = h.build_maxcut_hamiltonian(weight_matrix=weight_matrix)
    qaoa = q.QAOA_Build(repetitions=reps, sampler_seed=sampler_seed, optimizer_seed=optimizer_seed)
    most_likely_eigenstate, QAOA_Result = q.QAOA_Output(cost_hamiltonian, qaoa)
    expectation_value, ratio = q.approximation_ratio(most_likely_eigenstate, optimal_value, weight_matrix)
    return weight_matrix, optimal_bitstring, optimal_value, most_likely_eigenstate, ratio, expectation_value