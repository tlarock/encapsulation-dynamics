"""
    Compute and plot the number of DAG edges in the observed
    hypergraph compared to the series of random samples.
"""
import argparse
import sys
import pickle
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from encapsulation_dag import *
from utils import read_data, read_random_hyperedges

# Get inputs
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str,
                    help="Full path to dataset to randomize.")
parser.add_argument("--multiedges", action='store_true',
                    help="If given, include multiedges. Otherwise compute \
                    randomizations of simple graph.")
parser.add_argument("--random_start_num", type=int, default=0,
                    help="First random hypergraph to count DAG edges.")
parser.add_argument("--random_end_num", type=int, default=float("inf"),
                    help="Last random hypergraph to count edges from.")

args = parser.parse_args()
datapath = args.data_path

# setup for outputting random hypergraphs
datadir = "/".join(datapath.split("/")[0:-1])

multiedges = args.multiedges
first_randomization = args.random_start_num
last_randomization = args.random_end_num

# Read a hypergraph as a list of hyperedges
L = read_data(datapath, multiedges=multiedges)
if not multiedges:
    input_file = datadir + "/randomizations/random-simple-{}.txt"
else:
    input_file = datadir + "/randomizations/random-{}.txt"


# Construct hypergraph
dag_rw, nth_rw, he_map_rw = encapsulation_dag(L)

# DAG edges of observed data
observed_dag_edges = dag_rw.number_of_edges()

if not multiedges:
    output_file = datadir + "/randomizations/random-simple-"
else:
    output_file = datadir + "/randomizations/random-"

num_dag_edges = []
hypergraph_idx = first_randomization
while hypergraph_idx < last_randomization:
    print(hypergraph_idx)
    try:
        L = read_random_hyperedges(input_file.format(hypergraph_idx))
    except Exception as e:
        print(e)
        break
    dag_rw, nth_rw, he_map_rw = encapsulation_dag(L)
    num_dag_edges.append(dag_rw.number_of_edges())
    hypergraph_idx += 1

plt.figure()
plt.scatter([0], [observed_dag_edges], label="Observed")
plt.plot(list(range(len(num_dag_edges))), num_dag_edges, label="Randomized")
plt.xlabel(f"Steps")
plt.ylabel("# DAG Edges")
plt.legend()
if not multiedges:
    plt.savefig(datadir + f"/randomizations/simple_dag_edge_dist.pdf", dpi=200)
    with open(datadir + f"/randomizations/simple_dag_edge_dist.pickle", "wb") as fpickle:
        pickle.dump(num_dag_edges, fpickle)
else:
    plt.savefig(datadir + f"/randomizations/dag_edge_dist.pdf", dpi=200)
    with open(datadir + f"/randomizations/dag_edge_dist.pickle", "wb") as fpickle:
        pickle.dump(num_dag_edges, fpickle)
