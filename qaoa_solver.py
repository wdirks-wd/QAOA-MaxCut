from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Pauli
from qiskit.result import QuasiDistribution
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
import numpy as np
import classical as c

# Initializes QAOA.
def QAOA_Build(repetitions, sampler_seed, optimizer_seed):
    sampler = StatevectorSampler(seed = sampler_seed)
    algorithm_globals.random_seed = optimizer_seed
    optimizer = COBYLA()
    return QAOA(sampler, optimizer, reps = repetitions)

# Finds the bitstring with the highest probabilty to be received upon measurement.
def sample_most_likely(state_vector):
    return np.array([int(bit) for bit in max(state_vector.items(), key=lambda x: x[1])[0][::-1]])

# Returns the classical bitstring solution provided by QAOA and calculates that bitstrings associated objective value.
def QAOA_Output(cost_hamiltonian, qaoa):
    result = qaoa.compute_minimum_eigenvalue(-cost_hamiltonian)
    most_likely_eigenstate = sample_most_likely(result.eigenstate)
    return most_likely_eigenstate, result

# Calculates the approximation ratio between the QAOA objective value and the classical objective value.
def approximation_ratio(qaoa_bitstring, optimal_value, W):
    qaoa_value = c.weighted_objective_value(qaoa_bitstring, W)
    return qaoa_value, (qaoa_value / optimal_value)