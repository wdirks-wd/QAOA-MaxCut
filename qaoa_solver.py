from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Pauli
from qiskit.result import QuasiDistribution
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
import numpy as np
import classical as c
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import BitArray
from qiskit_aer.primitives import SamplerV2
from qaoa_training_pipeline.training.fixed_angle_conjecture import FixedAngleConjecture
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator



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

# Runs QAOA using SamplerV2 with Fixed Angle Conjecture and returns the expected cut value.
def run_qaoa_cut_experiment(graph, reps, sampler):
    cost_op = graph_to_operator(graph, pre_factor=-0.5)
    
    # Build circuit
    circuit = qaoa_ansatz(cost_op, reps=reps)
    circuit.measure_all()
    
    # Extract fixed angle parameters
    fa_params = FixedAngleConjecture(reps=reps).provide_params(cost_op=cost_op)
    
    # Execute circuit
    results = sampler.run([(circuit, fa_params["optimized_params"])]).result()
    joined_data = results[0].join_data()
    assert isinstance(joined_data, BitArray)
    
    # Process measurement bitstrings into weighted cut values
    counts = joined_data.get_counts()
    cut_values = [c.cut_size_from_graph(graph, b) for b in counts.keys()]
    frequencies = list(counts.values())
    
    # Return expected (average) cut value across all shots
    return float(np.average(cut_values, weights=frequencies))