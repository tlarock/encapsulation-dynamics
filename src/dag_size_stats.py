"""
    Script to compute the average number of DAG edges based
    on randomized data, to be shown in a plot. Split off from
    the larger compute_dag_stats script.
"""

import sys
import pickle
import networkx as nx
import numpy as np
from collections import Counter
sys.path.append("../src/")
from encapsulation_dag import *
from utils import read_data, read_random_hyperedges

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

data_dir = "../had_data/"
data_path_template = data_dir + "{}/randomizations/random-simple-"

datasets = ["coauth-MAG-History-full", "coauth-MAG-Geology-full",
            "coauth-DBLP-full", "NDC-Substances-Full"]

# For each dataset, which random DAG should we start
# and end our computation for?
dataset_first_dag = [40, 40, 40, 40]
dataset_last_dag = [101, 101, 101, 101]

for i in range(len(datasets)):
    dataset = datasets[i]
    print(dataset)
    # Read the observed DAG
    observed_path = data_dir + dataset + "/" + dataset + "-"
    obs_dag = read_dag(observed_path + "DAG.txt")
    data_path = data_path_template.format(dataset)
    first_dag = dataset_first_dag[i]
    last_dag = dataset_last_dag[i]

    dag_edges_dist = []
    for i in range(first_dag, last_dag):
        dag_input_file = data_path + f"{i}_dag.txt"
        dag = read_dag(dag_input_file)
        dag_edges_dist.append(dag.number_of_edges())

    print(f"Observed DAG Edges: {obs_dag.number_of_edges()} Mean of {last_dag-first_dag} randomizations: {np.mean(dag_edges_dist)} +/- {np.std(dag_edges_dist)}")
