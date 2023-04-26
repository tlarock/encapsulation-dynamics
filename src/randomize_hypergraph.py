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
from utils import read_data, write_hypergraph, read_hyperedges
sys.path.append("../../hypergraph/")
from hypergraph import hypergraph

# Get inputs
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str,
                    help="Full path to dataset to randomize.")
parser.add_argument("num_hypergraphs", type=int,
                    help="Number of randomizations to save.")
parser.add_argument("steps_per_iteration", type=int,
                    help="Number of Markov chain steps between hypergraphs.")
parser.add_argument("--multiedges", action='store_true',
                    help="If given, include multiedges. Otherwise compute \
                    randomizations of simple graph.")
parser.add_argument("--first_iter_steps", type=int, default=-1,
                    help="Use a different number of steps for the first \
                    iteration before collecting hypergraphs.")
parser.add_argument("--random_start_num", type=int, default=-1,
                    help="Start from precomputed hypergraph corresponding \
                    to this randomization number. Hypergraph must be stored in \
                    appropriate directory.")
parser.add_argument("--largest_cc", action="store_true",
                    help="If true, use dataset/dataset-cc.txt as observed hypergraph.")

args = parser.parse_args()
datapath = args.data_path

# setup for outputting random hypergraphs
datadir = "/".join(datapath.split("/")[0:-1])

num_hypergraphs = args.num_hypergraphs
steps_per_iter = args.steps_per_iteration
multiedges = args.multiedges
first_iter_steps = args.first_iter_steps
randomization_num = args.random_start_num
largest_cc = args.largest_cc

# Read a hypergraph as a list of hyperedges
if randomization_num < 0:
    if not largest_cc:
        L = read_data(datapath, multiedges=multiedges)
    else:
        L = read_hyperedges(datapath + "cc.txt")

    hypergraph_idx = 0
else:
    if not multiedges:
        input_file = datadir + "/randomizations/random-simple-{}.txt"
    else:
        input_file = datadir + "/randomizations/random-{}.txt"

    L = read_hyperedges(input_file.format(randomization_num))
    hypergraph_idx = randomization_num + 1
    num_hypergraphs += hypergraph_idx

# Construct hypergraph
G = hypergraph(L)

# First randomization
if first_iter_steps > 0:
    print(f"Running {first_iter_steps} steps as initialization of Markov chain.")
    G.MH(n_steps = first_iter_steps, label = 'vertex', detailed = True, n_clash = 1)

print(f"Computing {num_hypergraphs} randomizations with {steps_per_iter} steps each.")
G.MH(n_steps = steps_per_iter, label = 'vertex', detailed = True, n_clash = 1)
dag_rw, nth_rw, he_map_rw = encapsulation_dag(G.C)
num_dag_edges = []
num_dag_edges.append(dag_rw.number_of_edges())

if not multiedges:
    output_file = datadir + "/randomizations/random-simple-"
else:
    output_file = datadir + "/randomizations/random-"

# Randomize data
print(hypergraph_idx)
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
