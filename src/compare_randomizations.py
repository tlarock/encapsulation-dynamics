import numpy as np
import matplotlib.pyplot as plt
from encapsulation_dag import encapsulation_dag, overlap_dag, overlap_graph
from utils import read_data, read_hyperedges, largest_connected_component
from layer_randomization import layer_randomization

def print_observed_stats(cc):
    print("Computing encapsulation DAG...")
    obs_encap, _, _ = encapsulation_dag(cc)
    print("Number of encapsulation DAG edges in observed data: " + str(obs_encap.number_of_edges()))

    print("Computing overlap DAG...")
    obs_overlap_dag, _, _ = overlap_dag(cc)
    print("Number of overlap DAG in observed data: " + str(obs_overlap_dag.number_of_edges()))

    print("Computing overlap graph...")
    obs_overlap, _, _ = overlap_graph(cc)
    print("Number of overlap edges in observed data: " + str(obs_overlap.number_of_edges()))
    return obs_encap.number_of_edges(), obs_overlap_dag.number_of_edges(), obs_overlap.number_of_edges()


def print_random_stats(cc, obs_encap, obs_overdag, obs_overlap):
    print("Computing encapsulation DAG...")
    rnd_encap, _, _ = encapsulation_dag(cc)
    print(f"Number of encapsulation DAG edges in random data: {rnd_encap.number_of_edges()} %: {rnd_encap.number_of_edges() / obs_encap}")

    print("Computing overlap DAG...")
    rnd_overlap_dag, _, _ = overlap_dag(cc)
    print(f"Number of overlap DAG edges in random data: {rnd_overlap_dag.number_of_edges()} %: {rnd_overlap_dag.number_of_edges() / obs_overdag}")

    print("Computing overlap graph...")
    rnd_overlap, _, _ = overlap_graph(cc)
    print(f"Number of overlap edges in random data: {rnd_overlap.number_of_edges()} %: {rnd_overlap.number_of_edges() / obs_overlap}")
    return rnd_encap.number_of_edges(), rnd_overlap_dag.number_of_edges(), rnd_overlap.number_of_edges()



dataset_name = "coauth-MAG-History"
filename = f"../data/{dataset_name}/{dataset_name}-"
random_path = f"../data/{dataset_name}/randomizations/"
num_samples = 5

print("Reading hyperedges...")
hyperedges = read_data(filename, multiedges=False)
#hyperedges = read_hyperedges(filename)
print("Done.")

print("Computing largest connected component...")
cc = largest_connected_component(hyperedges, remove_single_nodes=True)
print("Done.")

obs_encap, obs_overdag, obs_overlap = print_observed_stats(cc)
obs_data = {
    "encap": obs_encap,
    "overdag": obs_overdag,
    "overlap": obs_overlap
}

print()

config_data = {
    "encap":[],
    "overdag":[],
    "overlap":[]
}

layer_data = {
    "encap":[],
    "overdag":[],
    "overlap":[]
}

for _ in range(num_samples):
    print("Reading configuration model data...")
    random = read_hyperedges(random_path + f"random-simple-nodetail-{_}.txt")
    cc = largest_connected_component(random, remove_single_nodes=True)
    encap, overdag, overlap = print_random_stats(cc, obs_encap, obs_overdag, obs_overlap)
    config_data["encap"].append(encap)
    config_data["overdag"].append(overdag)
    config_data["overlap"].append(overlap)
    print()

    print("Computing layer randomization...")
    random = layer_randomization(hyperedges)
    cc = largest_connected_component(random, remove_single_nodes=True)
    encap, overdag, overlap = print_random_stats(cc, obs_encap, obs_overdag, obs_overlap)
    layer_data["encap"].append(encap)
    layer_data["overdag"].append(overdag)
    layer_data["overlap"].append(overlap)
    print()

fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(7, 5))
for key in ["encap", "overdag", "overlap"]:
    conf = np.array(config_data[key]) / obs_data[key]
    axs[0][0].plot(list(range(conf.shape[0])), conf, label="Configuration " + key)
    axs[0][1].plot(list(range(conf.shape[0])), conf, label="Configuration " + key)
    axs[0][1].set(yscale="log")


    layer = np.array(layer_data[key]) / obs_data[key]
    axs[0][0].plot(list(range(layer.shape[0])), layer, label="Layer " + key)
    axs[0][1].plot(list(range(layer.shape[0])), layer, label="Layer " + key)
    axs[0][1].set(yscale="log")


axs[0][0].legend()
fig.tight_layout()
fig.savefig(f"comparison_{dataset_name}.pdf", dpi=150)
