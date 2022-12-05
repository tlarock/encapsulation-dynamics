import sys
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

def nodes_to_hyperedges(hyperedges):
    """
    Returns dictionary of nodes to hyperedges
    they participate in when hyperedges is a
    list-like of list-like edges.
    """
    nth = dict()
    he_map = dict()
    map_idx = 0
    for idx, he in enumerate(hyperedges):
        if he not in he_map:
            he_map[he] = map_idx
            map_idx += 1

        for node in he:
            if node not in nth:
                nth[node] = set()
            nth[node].add(he_map[he])
    return nth, he_map

def is_encapsulated(larger, smaller):
    return len(set(larger).intersection(set(smaller))) == len(smaller)

def encapsulation_dag(hyperedges):
    # Compute node to hyperedges
    nth, he_map = nodes_to_hyperedges(hyperedges)
    rev_map = {val:key for key,val in he_map.items()}
    # Construct the dag
    dag = nx.DiGraph()

    # Loop over hyperedges
    for he_idx, he in enumerate(hyperedges):
        dag.add_node(he)
        candidates = set()
        # Get all encapsulation candidate hyperedge ids
        for node in he:
            candidates.update([rev_map[i] for i in nth[node]])

        # for each candidate
        candidates_checked = set()
        for cand in candidates:
            if cand in candidates_checked:
                continue
            cand_idx = he_map[cand]
            if len(he) > len(cand):
                if is_encapsulated(he, cand):
                    dag.add_edge(he, cand)
            elif len(cand) > len(he):
                if is_encapsulated(cand, he):
                    dag.add_edge(cand, he)
    return dag, nth, he_map


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

def write_hypergraph(hypergraph, output_file):
    with open(output_file, "w") as fout:
        for he in hypergraph:
            if len(he) > 0:
                fout.write(','.join(list(map(str, he))) + "\n")
            else:
                print(f"Found empty hyperedge. Ignoring.")
    return None

# Get inputs
datapath = sys.argv[1]
# setup for outputinng random hypergraphs
datadir = "/".join(datapath.split("/")[0:-1])
num_hypergraphs = int(sys.argv[2])
steps_per_iter = int(sys.argv[3])
from_random = True
last_random = 99
remove_multiedges = False

if not from_random:
    # Read a hypergraph as a list of hyperedges
    L = read_data(datapath)

    if remove_multiedges:
        tupL = set([tuple(c) for c in L])
        L = [list(c) for c in tupL]

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
