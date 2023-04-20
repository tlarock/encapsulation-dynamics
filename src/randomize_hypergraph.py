"""
    Uses Phil Chodrow's python code for hypergraph configuration model
    to randomize a hypergraph. Keeps track of how many encapsulation
    relationships are in the randomized data as the target variable
    for Markov chain convergence.
"""
import argparse
import sys
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from encapsulation_dag import *
from utils import read_data, write_hypergraph
sys.path.append("../../hypergraph/")
from hypergraph import hypergraph

# Get inputs
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="Full path to dataset to randomize.")
parser.add_argument("num_hypergraphs", type=int, help="Number of randomizations to save.")
parser.add_argument("steps_per_iteration", type=int, help="Number of Markov chain steps between hypergraphs.")
parser.add_argument("--multiedges", action='store_true', help="If given, include multiedges. Otherwise compute randomizations of simple graph.")

args = parser.parse_args()
datapath = args.data_path

# setup for outputting random hypergraphs
datadir = "/".join(datapath.split("/")[0:-1])

num_hypergraphs = args.num_hypergraphs
steps_per_iter = args.steps_per_iteration
multiedges = args.multiedges

# Read a hypergraph as a list of hyperedges
L = read_data(datapath, multiedges=multiedges)

# Construct hypergraph
G = hypergraph(L)
hypergraph_idx = 0

# First randomization
print("Initial randomization")
G.MH(n_steps = steps_per_iter, label = 'vertex', detailed = True, n_clash = 1)
dag_rw, nth_rw, he_map_rw = encapsulation_dag(G.C)
num_dag_edges = []
num_dag_edges.append(dag_rw.number_of_edges())

if not multiedges:
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
if not multiedges:
    plt.savefig(datadir + f"/randomizations/simple_dag_edge_dist.pdf", dpi=200)
else:
    plt.savefig(datadir + f"/randomizations/dag_edge_dist.pdf", dpi=200)
