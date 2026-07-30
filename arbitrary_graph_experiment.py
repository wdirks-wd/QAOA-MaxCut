import numpy as np
import matplotlib.pyplot as plt
import trial as t
import statistics as s
import visualization as v

# Experimental Setup
output_file_path = "/Users/williamdirks/QuantumComputingResearch/"

master_seed = 1
rng = np.random.default_rng(master_seed)
graph_seed = int(rng.integers(1e9))

num_nodes = 4
graph_degree = 3
num_reps = 10
num_sigmas = 30
sigmas = np.linspace(0,10,num_sigmas)

# Create Output File Structure
(node_folder, 
 node_plots_folder,
 node_data_folder,
 reps_file_paths, 
 stats_file_paths, 
 hist_file_paths, 
 scatter_file_paths
 )=v.create_experiment_folder(
                        output_file_path=output_file_path, 
                        master_seed=master_seed, 
                        graph_degree=graph_degree, 
                        num_nodes=num_nodes, 
                        num_reps=num_reps, 
                        num_sigmas=num_sigmas
                        )

# Initialize Persistent Matrices 
UW_ratios=[]
W_ratios_matrix=[]
W_mean_matrix=[]
W_std_matrix=[]
W_stderr_matrix=[]

for i in range(num_reps):

    # Select File Paths for current data to be output to
    current_depth_file_path = reps_file_paths[i]
    current_stats_file_path = stats_file_paths[i]
    current_hist_file_path = hist_file_paths[i]
    current_scatter_file_path = scatter_file_paths[i]

    # Initialize temp arrays for given sigma results
    W_ratios=[]
    W_stats=[]
    W_mean=[]
    W_std=[]
    W_stderr=[]

    # Select seeds for specific trial run, these remain fixed for all sigmas of given depth
    perturbation_seed = rng.integers(1e9)
    sampler_seed = rng.integers(1e9)
    optimizer_seed = rng.integers(1e9)

    # For each depth, find the approximation ratio for the unweighted underlying graph.
    (
    UW_edges, 
    UW_optimal_bitstring, 
    UW_optimal_value, 
    UW_most_likely_eigenstate, 
    UW_ratio, 
    UW_expectation_value
    ) = t.run_unweighted_trial(
                            num_nodes=num_nodes, 
                            reps=(i+1),
                            graph_seed=graph_seed, 
                            degree=graph_degree,
                            sampler_seed=sampler_seed, 
                            optimizer_seed=optimizer_seed
                            )   

    UW_ratios.append(UW_ratio)

    # For each sigma value, run a single QAOA trial on the underlying graph with the edge weights being sampled 
    # from a normal curve centered about zero with a standard deviation of [sigma]. Returns an approximation ratio for each sigma.
    for sigma in sigmas:
        (
        W_edges, 
        W_optimal_bitstring, 
        W_optimal_value, 
        W_most_likely_eigenstate, 
        W_ratio, 
        W_expectation_value,
        ) = t.run_weighted_trial(
                                num_nodes=num_nodes, 
                                reps=(i+1),
                                sigma=sigma,
                                perturbation_seed=perturbation_seed,
                                graph_seed=graph_seed, 
                                degree=graph_degree,
                                sampler_seed=sampler_seed, 
                                optimizer_seed=optimizer_seed
                                )

        # Save AR values for current sigma to W_Ratios.
        W_ratios.append(W_ratio)

    # After all AR Values have been calculated for all sigmas, convert W_Ratios to W_ratio_array.
    W_ratio_array = np.array(W_ratios)

    # Use W_ratio_array to calculate the mean, std, and stderr across all sigmas in current depth.
    W_ratios_mean = W_ratio_array.mean(axis=0)
    W_ratios_std = W_ratio_array.std(axis=0)
    W_ratios_stderr = W_ratios_std / np.sqrt(num_sigmas)

    # Append these statistical values to array which is capable of being written to csv.
    W_mean.append(W_ratios_mean)
    W_std.append(W_ratios_std)
    W_stderr.append(W_ratios_stderr)

    # Save W_ratios and statitsics to current depth folder/stats folder.
    np.savetxt(f"{current_depth_file_path}/Weighted Ratios.csv", W_ratios, delimiter=",")
    np.savetxt(f"{current_stats_file_path}/Weighted Ratio Average.csv", W_mean, delimiter=",")   
    np.savetxt(f"{current_stats_file_path}/Weighted Ratio Std.csv", W_std, delimiter=",")   
    np.savetxt(f"{current_stats_file_path}/Weighted Ratio Stderr.csv", W_stderr, delimiter=",")  

    # Save U and UW Ratios, and all statistics to their persistent, global arrays.
    UW_ratio_array = np.array(UW_ratios)
    W_ratios_matrix.append(W_ratios)
    W_ratios_array = np.array(W_ratios_matrix)


    W_mean_matrix.append(W_mean)
    W_std_matrix.append(W_std)
    W_stderr_matrix.append(W_stderr)

    # Convert Global statistical lists to numpy arrays.
    W_mean_array = np.array(W_mean_matrix)
    W_std_array = np.array(W_std_matrix)
    W_stderr_array = np.array(W_stderr_matrix)

    # Create dataframes, then create histograms and scatterplots for each depth.
    single_depth_weighted_ratio_data_frame = v.csv_2_df(f"{current_depth_file_path}/Weighted Ratios.csv")
    single_depth_weighted_ratio_mean_data_frame = v.csv_2_df(f"{current_stats_file_path}/Weighted Ratio Average.csv")
    v.hist(single_depth_weighted_ratio_data_frame, single_depth_weighted_ratio_mean_data_frame, 20, f"{current_hist_file_path}/", f"Depth = {i+1} Histogram", "Approximation Ratio", "Perturbation Standard Deviation")
    v.scatter(single_depth_weighted_ratio_data_frame, f"{current_scatter_file_path}/", f"Depth = {i+1} Scatterplot", "Perturbation Standard Deviation", "Approximation Ratio")

# Save the global data which is persistent across all depths to the node folder.
np.savetxt(f"{node_data_folder}/Unweighted Ratios.csv", UW_ratio_array, delimiter=",")
np.savetxt(f"{node_data_folder}/Weighted Ratios.csv", W_ratios_matrix, delimiter=",")
np.savetxt(f"{node_data_folder}/Weighted Ratios Averages.csv", W_mean_array, delimiter=",")   
np.savetxt(f"{node_data_folder}/Weighted Ratios Stds.csv", W_std_array, delimiter=",")   
np.savetxt(f"{node_data_folder}/Weighted Ratios Stderrs.csv", W_stderr_array, delimiter=",")  

# Load the data within csv's to be used in plot production into pandas dtaframes.
unweighted_ratio_data_frame = v.csv_2_df(f"{node_data_folder}/Unweighted Ratios.csv")
weighted_ratio_data_frame = v.csv_2_df(f"{node_data_folder}/Weighted Ratios.csv")
weighted_ratio_mean_data_frame = v.csv_2_df(f"{node_data_folder}/Weighted Ratios Averages.csv")
weighted_ratio_std_data_frame = v.csv_2_df(f"{node_data_folder}/Weighted Ratios Stds.csv")
weighted_ratio_stderr_data_frame = v.csv_2_df(f"{node_data_folder}/Weighted Ratios Stderrs.csv")

# Generate a scatterplot for the unweighted approximation ratios vs. the QAOA circuit depth.
v.simple_scatter(unweighted_ratio_data_frame, f"{node_plots_folder}/", f"Unweighted AR vs. Depth Scatterplot", "Depth", "Approximation Ratio")

# Generate scatterplots of the weighted approximation ratios for each sigma vs. the QAOA circuit depth.
for i in range(num_sigmas):
    v.simple_scatter(weighted_ratio_data_frame.iloc[:,i], f"{node_plots_folder}/", f"Weighted AR vs. Depth Scatterplot Sigma # {i+1}", "Depth", "Approximation Ratio")

# Generate a boxplot which shows the spread of the approximation ratios calculated for each sigma within a given circuit depth against depth.
v.boxplotter(weighted_ratio_data_frame.T, node_plots_folder, "Approximation Ratio vs. Depth", "Circuit Depth", "Approximation Ratio")

# Plot average AR over sigmas, std of AR wrt sigmas, and unweighted AR values all vs. depth
plt.plot(unweighted_ratio_data_frame.index + 1, unweighted_ratio_data_frame, alpha=0.7, c='red', label='Unweighted Ratio')
plt.plot(weighted_ratio_mean_data_frame.index + 1, weighted_ratio_mean_data_frame, alpha=0.7, c='green', label='Weighted Ratio Average')
plt.plot(weighted_ratio_std_data_frame.index + 1, weighted_ratio_std_data_frame, alpha=0.7, c='blue', label='Weighted Ratio Standard Deviation')
plt.legend()
plt.title("Statistics vs. Depth")
plt.xlabel("Circuit Depth")
plt.ylabel("Approximation Ratio")
plt.savefig(f"{node_plots_folder}/Statistics Vs. Depth Plot.pdf", dpi=300, bbox_inches='tight')
plt.close()