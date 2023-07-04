import sys
import pickle
import numpy as np
from encapsulation_dag import encapsulation_dag, overlap_dag, overlap_graph
from utils import read_data, read_hyperedges, largest_connected_component
from layer_randomization import layer_randomization

def print_observed_stats(cc):
    print("Computing encapsulation DAG...")
    obs_encap, _, _ = encapsulation_dag(cc)
    print("Number of encapsulation DAG edges in observed data: " + str(obs_encap.number_of_edges()))
    encap_dag_edges = obs_encap.number_of_edges()
    del obs_encap
    print("Computing overlap DAG...")
    obs_overlap_dag, _, _ = overlap_dag(cc)
    print("Number of overlap DAG in observed data: " + str(obs_overlap_dag.number_of_edges()))
    overlap_dag_edges = obs_overlap_dag.number_of_edges()
    del obs_overlap_dag
    print("Computing overlap graph...")
    obs_overlap, _, _ = overlap_graph(cc, normalize_weight=False)
    obs_overlap_edges = obs_overlap.number_of_edges()
    sum_of_weights = sum([data["weight"] for _,_, data in obs_overlap.edges(data=True)])
    print("Number of overlap edges in observed data: " + str(obs_overlap.number_of_edges()))
    del obs_overlap
    return encap_dag_edges, overlap_dag_edges, obs_overlap_edges, sum_of_weights


def print_random_stats(cc, obs_encap, obs_overdag, obs_overlap):
    print("Computing encapsulation DAG...")
    rnd_encap, _, _ = encapsulation_dag(cc)
    print(f"Number of encapsulation DAG edges in random data: {rnd_encap.number_of_edges()} %: {rnd_encap.number_of_edges() / obs_encap}")

    print("Computing overlap DAG...")
    rnd_overlap_dag, _, _ = overlap_dag(cc)
    print(f"Number of overlap DAG edges in random data: {rnd_overlap_dag.number_of_edges()} %: {rnd_overlap_dag.number_of_edges() / obs_overdag}")

    print("Computing overlap graph...")
    rnd_overlap, _, _ = overlap_graph(cc, normalize_weight=False)
    sum_of_weights = sum([data["weight"] for _,_, data in rnd_overlap.edges(data=True)])
    print(f"Number of overlap edges in random data: {rnd_overlap.number_of_edges()} %: {rnd_overlap.number_of_edges() / obs_overlap}")
    return rnd_encap.number_of_edges(), rnd_overlap_dag.number_of_edges(), rnd_overlap.number_of_edges(), sum_of_weights

dataset_name = sys.argv[1]
num_samples = int(sys.argv[2])

print(dataset_name)
#filename = f"../data/{dataset_name}/{dataset_name}-"
filename = f"/data/math-networks/math1764/hypergraphs/{dataset_name}/{dataset_name}-"
print("Reading hyperedges...")
obs_hyperedges = read_data(filename, multiedges=False)
print("Done.")

print("Computing largest connected component...")
obs_cc = largest_connected_component(obs_hyperedges, remove_single_nodes=False)
print("Done.")
obs_encap, obs_overdag, obs_overlap, obs_overlap_sum = print_observed_stats(obs_cc)

obs_data = dict()
obs_data[dataset_name] = {
    "encap": obs_encap,
    "overdag": obs_overdag,
    "overlap": obs_overlap,
    "overlap_sum": obs_overlap_sum
}

layer_data = dict()
layer_data[dataset_name] = {
    "encap":[],
    "overdag":[],
    "overlap":[],
    "overlap_sum":[]
}

for i in range(num_samples):
    print(f"Computing layer randomization {i}...")
    random = layer_randomization(obs_hyperedges)
    cc = largest_connected_component(random, remove_single_nodes=True)
    encap, overdag, overlap, overlap_sum = print_random_stats(cc, obs_encap, obs_overdag, obs_overlap)
    layer_data[dataset_name]["encap"].append(encap)
    layer_data[dataset_name]["overdag"].append(overdag)
    layer_data[dataset_name]["overlap"].append(overlap)
    layer_data[dataset_name]["overlap_sum"].append(overlap_sum)
    print()

#with open(f"../results/{dataset_name}/randomization_comparison.pickle", "wb") as fpickle:
with open(f"/data/math-networks/math1764/hypergraphs/results/{dataset_name}/randomization_comparison.pickle", "wb") as fpickle:
    pickle.dump((obs_data[dataset_name], layer_data[dataset_name]), fpickle)
