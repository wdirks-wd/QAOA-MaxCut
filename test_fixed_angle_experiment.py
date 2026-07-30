import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import classical as c
import qaoa_solver as q
import visualization as v
from qiskit_aer.primitives import SamplerV2

degree = 3              
num_nodes = 6            
max_depth = 11 
num_samples = 10           
num_weighted_experiments = 5    
sigma_constant = 0.1        

sampler = SamplerV2()

base_graph = nx.random_regular_graph(degree, num_nodes)
for u, v in base_graph.edges():
    base_graph[u][v]["weight"] = 1.0

adj_matrix = nx.to_numpy_array(base_graph, weight='weight')
opt_bitstring, opt_value = c.classical_max_cut(adj_matrix)

df_array = []
mean_array = []
std_array = []
stderr_array = []

for depth in range(max_depth):

    # Prepare results storage dictionary
    results_data = {}

    # Unweighted Base Experiment
    print(f"({depth+1}) Running Unweighted Experiment...")

    # Runs multiple times to account for quantum measurement variance
    results_data["Unweighted"] = [q.run_qaoa_cut_experiment(base_graph, depth+1, sampler) for _ in range(num_samples)]/opt_value

    # Weighted Sub-Experiments with Perturbations
    for k in range(1, num_weighted_experiments + 1):
        sigma = sigma_constant * k
        col_name = f"σ = {sigma:.3f}"
        print(f"Running Sub-Experiment {k}/{num_weighted_experiments} (sigma = {sigma:.4f})...")
    
        exp_cut_values = []
        for _ in range(num_samples):

            # For each sample, copy graph structure and perturb weights by N(0, sigma^2)
            # This allows us to average the approximation ratio over many samples to reduce
            # the impact of extreme perturbations, showing generally how a given std 
            # affects the convergence of the qaoa onto the exact maxcut value.

            perturbed_graph = base_graph.copy()
            for u, v in perturbed_graph.edges():
                noise = abs(np.random.normal(0, sigma))
                perturbed_graph[u][v]["weight"] += noise

            # Calculate exact max cut value for perturbed grtaph classically
            weight_matrix = nx.to_numpy_array(perturbed_graph, weight='weight')
            w_opt_bitstring, w_opt_value = c.classical_max_cut(weight_matrix)
            
            # Run QAOA on the perturbed graph
            avg_cut = q.run_qaoa_cut_experiment(perturbed_graph, depth+1, sampler)
            exp_cut_values.append(avg_cut/w_opt_value)

        # Store experimental results in array specific to current depth
        results_data[col_name] = exp_cut_values

    # Convert arrays of results into pandas dataframes to reduce complexity
    df_results = pd.DataFrame(results_data)
    df_results.index.name = "sample_number"

    # Calculate the Mean, STD, and STDERR across all samples for each sigma.
    df_mean = df_results.mean(numeric_only=True)
    df_std = df_results.std(numeric_only=True)
    df_stderr = df_std/np.sqrt(num_samples)
    
    df_array.append(df_results)
    mean_array.append(df_mean)
    std_array.append(df_std)
    stderr_array.append(df_stderr)

# Take the results for each depth and create mean, std, and stderr dataframes for the entire experiment with dimensions ()
df_means = pd.concat(mean_array, axis = 1)
df_stds = pd.concat(std_array, axis = 1)
df_stderrs = pd.concat(stderr_array, axis = 1)

print(df_means)

depths = np.arange(1, max_depth + 1)

fig, ax = plt.subplots(figsize=(8, 6))

for index in df_means.index:
    y_means = df_means.loc[index].values
    y_errs = df_stderrs.loc[index].values
    
    ax.errorbar(
        x=depths,
        y=y_means,
        yerr=y_errs,
        fmt='-o',
        capsize=4,
        label=index
    )

ax.set_xlabel("QAOA Circuit Depth (p)")
ax.set_ylabel("Approximation Ratio")
ax.set_title(f"QAOA Approximation Ratio vs Depth for {num_nodes} Nodes, {num_samples} Samples")
ax.set_xticks(depths)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()