import sys
import pickle
import networkx as nx
import numpy as np
from collections import Counter
sys.path.append("../src/")
from utils import read_data
from encapsulation_dag import *
sys.path.append("../../hypergraph/")
from hypergraph import hypergraph
from layer_randomization import layer_randomization
sys.path.append("../../tr-dag-cycles/")
from cycle_utilities import tr

def compute_dag_heights(dag):
    # transitiviely reduce the DAG
    dag_tr = tr(dag.copy())
    heights = []
    # Get all of the nodes that have no in-degree and non-zero out-degree
    root_nodes = [node for node in dag_tr.nodes() if dag_tr.out_degree(node) > 0 and dag_tr.in_degree(node) == 0]
    for source in root_nodes:
        sp_dict = nx.single_source_shortest_path_length(dag_tr, source)
        for target in sp_dict:
            if source == target:
                continue
            heights.append(sp_dict[target])
    return heights

data_dir = "../data/"
dataset = sys.argv[1]
num_samples = 10

observed_path = data_dir + dataset + "/" + dataset + "-"
print("Reading hyperedges.")
hyperedges = read_data(observed_path, multiedges=False)

print("Computing observed dag.")
obs_dag, obs_nth, obs_he_map = encapsulation_dag(hyperedges)

print(f"Dag edges: {obs_dag.number_of_edges()}")

# Observed
dag = obs_dag
print("Computing observed dag heights.")
heights = compute_dag_heights(dag.copy())
height_output_file = data_dir + dataset + f"/{dataset}_dag_heights.txt"
with open(height_output_file, "w") as fout:
    fout.write(",".join(map(str,heights)))
print(f"Observed dag average height: {np.mean(heights)}")

# Random
heights = []
for _ in range(num_samples):
    print("Computing layer randomization.")
    random_hyperedges = layer_randomization(hyperedges)
    #### Heights ####
    print("Computing random dag.")
    dag, _, _ = encapsulation_dag(random_hyperedges)
    print(f"Random dag has {dag.number_of_edges()} edges.")
    print("Computing random dag heights.")
    heights.append(compute_dag_heights(dag.copy()))
    print(f"Random dag average height: {np.mean(heights[-1])}")

height_output_file = data_dir + dataset + f"/{dataset}_layer_randomization_dag_heights.txt"
with open(height_output_file, "w") as fout:
    for sample_heights in heights:
        fout.write(",".join(map(str,sample_heights)) + "\n")
