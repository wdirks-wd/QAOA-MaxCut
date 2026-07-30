import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

# Creates a folder given an input filepath for the folder to be stored. Returns the filepath of the folder.
def create_folder(parent_directory, folder_name):
    full_path = Path(parent_directory) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    
    return full_path

def create_fa_exp_folder(output_file_path, degree, num_nodes, num_reps, seed):
    node_max = 2*num_nodes + 2
    exp_folder = create_folder(output_file_path, f"{degree}-Regular Graph, Max Nodes = {node_max}, Max Depth = {num_reps}, Master Seed = {seed}")
    return exp_folder

# Creates the entire file structure for the experimental data to be stored. Returns the file paths of each folder.
def create_experiment_folder(output_file_path, master_seed, graph_degree, num_nodes, num_reps, num_sigmas):
    node_folder = create_folder(output_file_path, f"Nodes = {num_nodes}, {graph_degree} Regular, Max Depth = {num_reps}, Sigma Count = {num_sigmas}, Master Seed = {master_seed}")
    node_plots_folder = create_folder(node_folder, "Plots")
    node_data_folder = create_folder(node_folder, "Global Data")

    reps_file_paths = []
    stats_file_paths =[]
    hist_file_paths = []
    scatter_file_paths = []

    for i in range(num_reps):
        reps_folder = create_folder(node_folder, f"Depth = {i+1}")
        stats_folder = create_folder(reps_folder, f"Depth = {i+1} Statistics")
        hist_folder = create_folder(reps_folder, f"Depth = {i+1} Histograms")
        scatter_folder = create_folder(reps_folder, f"Depth = {i+1} Scatterplots")
        reps_file_paths.append(reps_folder)
        stats_file_paths.append(stats_folder)
        hist_file_paths.append(hist_folder)
        scatter_file_paths.append(scatter_folder)

    return node_folder, node_plots_folder, node_data_folder, reps_file_paths, stats_file_paths, hist_file_paths, scatter_file_paths

# Converts a csv file to a pandas dataframe.
def csv_2_df(main_file_path):
    df = pd.read_csv(main_file_path, header=None)
    return df

# Creates a histogram from a pandas dataframe.
def hist(df_main, df_mean, bin_count, output_file_path, title, x_axis, y_axis):

    column_count = df_main.shape[1]
    global_min = df_main.min().min()
    global_max = df_main.max().max()
    custom_bins = np.linspace(global_min, global_max, bin_count + 1)

    for i in range(column_count):
        fig, ax = plt.subplots()
        mean = df_mean.iloc[i,0]
        df_main.iloc[:,i].hist(bins=custom_bins, edgecolor='black', color='skyblue', ax=ax)
        ax.set_xlim(global_min, global_max)
        ax.axvline(mean, color='darkgreen', linestyle='-.', linewidth=2, label=f'Mean ({mean:.4f})')
        ax.set_title(f"{title}")
        ax.set_xlabel(f'{x_axis}, Bin Size = {1/bin_count}')
        ax.set_ylabel(f'{y_axis}')
        ax.legend()
        plt.savefig(f"{output_file_path}{title} Histogram.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    return

# Creates a scatterplot from a pandas dataframe.
def scatter(df, output_file_path, title, x_axis, y_axis):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col_name in num_cols:
        fig, ax = plt.subplots()
        ax.scatter(df.index + 1, df[col_name], alpha=0.7, edgecolors='none', s=25)
        ax.set_title(f"{title}", fontsize=12)
        ax.set_xlabel(f"{x_axis}", fontsize=10)
        ax.set_ylabel(f"{y_axis}", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{output_file_path}{title} Scatterplot.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

# Creates a scatterplot from a pandas dataframe without column names.
def simple_scatter(df, output_file_path, title, x_axis, y_axis):
    fig, ax = plt.subplots()
    ax.scatter(df.index + 1, df, alpha=0.7, edgecolors='none', s=25)
    ax.set_title(f"{title} Scatterplot", fontsize=12)
    ax.set_xlabel(f"{x_axis}", fontsize=10)
    ax.set_ylabel(f"{y_axis}", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{output_file_path}{title} Scatterplot.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

def boxplotter(df, output_file_path, title, x_axis, y_axis):
    plt.boxplot(df)
    plt.title(f"{title}")
    plt.xlabel(f"{x_axis}")
    plt.ylabel(f"{y_axis}")
    plt.savefig(f"{output_file_path}/{title} Boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    return

# Produces image of graph from the input weight matrix.
def generate_graph(weight_matrix, graph_seed):
    graph = nx.from_numpy_array(weight_matrix)
    layout = nx.random_layout(graph, seed=graph_seed)
    nx.draw(graph, layout, with_labels=True)
    labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=labels);
    plt.show()

    return