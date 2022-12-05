import sys
import pickle
import networkx as nx
import numpy as np
from collections import Counter
sys.path.append("../src/")
from utils import read_data, read_random_hyperedges
from encapsulation_dag import *

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
dataset = sys.argv[1]
parallel_edges = True
read_dags = False
num_randomizations = 50

observed_path = data_dir + dataset + "/" + dataset + "-"
if not read_dags:
    L = read_data(observed_path)
    G = hypergraph(L)
    obs_dag, obs_nth, obs_he_map = encapsulation_dag(G.C)
    write_dag(obs_dag, observed_path + "DAG.txt")
else:
    obs_dag = read_dag(observed_path + "DAG.txt")

print(f"Dag edges: {obs_dag.number_of_edges()}")

# Get overlap dists from observed dag
overlap_dists = get_overlap_dists(obs_dag, binomial_norm=True)
# Observed
dag = obs_dag
heights = compute_dag_heights(dag.copy())
height_output_file = data_dir + dataset + f"/{dataset}_dag_heights.txt"
with open(height_output_file, "w") as fout:
    fout.write(",".join(map(str,heights)))

if not parallel_edges:
    data_path = data_dir + dataset + "/randomizations/random-simple-"
else:
    data_path = data_dir + dataset + "/randomizations/random-"


dag_edges_dist = []
n_range = [5, 4, 3, 2]
dag_overlap_dists = dict()
for n in n_range:
    dag_overlap_dists[n] = dict()
    for m in overlap_dists[n]:
        dag_overlap_dists[n][m] = np.zeros((num_randomizations, math.comb(n,m)))

for i in range(0, num_randomizations):
    print(i)
    if read_dags:
        dag_input_file = data_path + f"{i}_dag.txt"
        dag = read_dag(dag_input_file)
    else:
        random_hyperedges = read_random_hyperedges(data_path + f"{i}.txt")
        dag, nth, he_map = encapsulation_dag(random_hyperedges)
        output_file = data_path + f"{i}_dag.txt"
        write_dag(dag, output_file)

    dag_edges_dist.append(dag.number_of_edges())
    curr_od = get_overlap_dists(dag, binomial_norm=True)
    for n in n_range:
        for m in range(n-1, 0, -1):
            if n in curr_od and m in curr_od[n]:
                d = dict(Counter(curr_od[n][m]))
                dkeys = sorted(d.keys())
                for idx in range(len(dkeys)):
                    dag_overlap_dists[n][m][i, idx] = d[dkeys[idx]]

    #### Heights ####
    heights = compute_dag_heights(dag.copy())
    height_output_file = data_path + f"{i}_dag_heights.txt"
    with open(height_output_file, "w") as fout:
        fout.write(",".join(map(str,heights)))


print(f"Mean: {np.mean(dag_edges_dist)} +/- {np.std(dag_edges_dist)}")
if not parallel_edges:
    outfile = data_dir + dataset + "/randomizations/random-simple_overlaps.pickle"
else:
    outfile = data_dir + dataset + "/randomizations/random_overlaps.pickle"

with open(outfile, "wb") as fpickle:
        pickle.dump(dag_overlap_dists, fpickle)
print("Dumped dag_overlap_dists")
