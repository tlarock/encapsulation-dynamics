"""
    Uses Phil Chodrow's python code for hypergraph configuration model
    to randomize a hypergraph. Keeps track of how many encapsulation
    relationships are in the randomized data as the target variable
    for Markov chain convergence.
"""

import sys
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from encapsulation_dag import *
from utils import read_data, write_hypergraph
sys.path.append("../../hypergraph/")
from hypergraph import hypergraph
# Get inputs
datapath = sys.argv[1]
# setup for outputinng random hypergraphs
datadir = "/".join(datapath.split("/")[0:-1])
num_hypergraphs = int(sys.argv[2])
steps_per_iter = int(sys.argv[3])
from_random = False
last_random = 0
remove_multiedges = True

if not from_random:
    # Read a hypergraph as a list of hyperedges
    L = read_data(datapath, multiedges=not remove_multiedges)

    # Construct hypergraph
    G = hypergraph(L)

    hypergraph_idx = 0
else:
    # Construct the path to the random data
    path_to_random = datadir + "/randomizations/random-" + f"{last_random}.txt"
    L = read_data(datapath)
    G = hypergraph(L)
    hypergraph_idx = last_random
    num_hypergraphs = num_hypergraphs + hypergraph_idx

# First randomization
print("Initial randomization")
G.MH(n_steps = steps_per_iter, label = 'vertex', detailed = True, n_clash = 1)
dag_rw, nth_rw, he_map_rw = encapsulation_dag(G.C)
num_dag_edges = []
num_dag_edges.append(dag_rw.number_of_edges())

if remove_multiedges:
    output_file = datadir + "/randomizations/random-simple-"
else:
    output_file = datadir + "/randomizations/random-"

# Randomize data
write_hypergraph(G.C, output_file + f"{hypergraph_idx}.txt")
while hypergraph_idx < num_hypergraphs:
    hypergraph_idx += 1
    print(hypergraph_idx)
    G.MH(n_steps = steps_per_iter, label = 'vertex', detailed = True, n_clash = 1)
    write_hypergraph(G.C, output_file + f"{hypergraph_idx}.txt")
    dag_rw, nth_rw, he_map_rw = encapsulation_dag(G.C)
    num_dag_edges.append(dag_rw.number_of_edges())


plt.figure()
plt.plot(list(range(len(num_dag_edges))), num_dag_edges)
plt.xlabel(f"Steps (in {steps_per_iter})")
plt.ylabel("# DAG Edges")
if remove_multiedges:
    plt.savefig(datadir + f"/randomizations/simple_dag_edge_dist.pdf", dpi=200)
else:
    plt.savefig(datadir + f"/randomizations/dag_edge_dist.pdf", dpi=200)
