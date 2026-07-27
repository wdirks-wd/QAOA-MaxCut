import numpy as np
from qiskit.quantum_info import SparsePauliOp

# Constructs the max cut hamiltonian from the graph's weight matrix.
def build_maxcut_hamiltonian(weight_matrix):
    # Constructs cost hamiltonian
    num_nodes = weight_matrix.shape[0]
    cost_paulis = []

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if weight_matrix[i,j] != 0:
                cost_paulis.append(("ZZ", [i, j], -0.5*weight_matrix[i, j]))
                cost_paulis.append(("I"*num_nodes, list(range(num_nodes)), 0.5*weight_matrix[i, j]))
    return SparsePauliOp.from_sparse_list(cost_paulis, num_qubits=num_nodes)

# Constructs the X mixer hamiltonian by appending the Pauli X gate to each qubit. 
def build_x_mixer_hamiltonian(num_nodes):
    mixer_paulis = []

    for i in range(num_nodes):
        mixer_paulis.append(("X", [i], 1.0))

    return SparsePauliOp.from_sparse_list(mixer_paulis, num_qubits=num_nodes)