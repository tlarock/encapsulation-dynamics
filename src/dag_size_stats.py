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

sys.path.append("../../tr-dag-cycles/")
from cycle_utilities import tr

sys.path.append("../../hypergraph/")
from hypergraph import *
from read import *
from scipy.stats import rankdata
# Slightly modifying Phil's read data frunction for arbitrary paths
def read_data(path, t_min = None, t_max = None):
    # read in the data
    nverts = np.array([int(f.rstrip('\n')) for f in open(path + 'nverts.txt')])
    times = np.array([float(f.rstrip('\n')) for f in open(path + 'times.txt')])
    simplices = np.array([int(f.rstrip('\n')) for f in open(path + 'simplices.txt')])

    times_ex = np.repeat(times, nverts)

    # time filtering

    t_ix = np.repeat(True, len(times))
    if t_min is not None:
        simplices = simplices[times_ex >= t_min]
        nverts = nverts[times >= t_min]
    if t_max is not None:
        simplices = simplices[times_ex <= t_max]
        nverts = nverts[times <= t_max]

    # relabel nodes: consecutive integers from 0 to n-1

    unique = np.unique(simplices)
    mapper = {unique[i] : i for i in range(len(unique))}
    simplices = np.array([mapper[s] for s in simplices])

    # format as list of lists

    l = np.split(simplices, np.cumsum(nverts))
    C = [list(c) for c in l]
    return(C)

def read_random_hyperedges(filename):
    hyperedges = []
    with open(filename, "r") as fin:
        for line in fin:
            s = list(map(int, line.strip().split(',')))
            hyperedges.append(tuple(s))
    return hyperedges

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
