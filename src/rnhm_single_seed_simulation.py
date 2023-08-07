import numpy as np
from collections import defaultdict

from encapsulation_dag import encapsulation_dag, overlap_dag, get_overlap_dists
from utils import check_hyperedges_connectivity

import xgi
from update_rules import *
from selection_rules import *
from simulation import *
from seed_functions import *
from plot_simulation_results import *
from random_nested_hypergraph import random_nested_hypergraph
import matplotlib_defaults
import matplotlib.pyplot as plt


# Simulation parameters
configuration = {
    "steps": 5,
    "active_threshold": "all",
    "num_simulations": 50
}

configuration["num_hypergraphs"] = 50

configuration["seeding_strategy"] = "edge"
#configuration["seed_function"] = "uniform"
configuration["seed_function"] = smallest_first_seed
configuration["selection_name"] = "simultaneous"
configuration["selection_function"] = None

configuration["update_name"] = "encapsulation-immediate"
configuration["encapsulation_update"] = True
configuration["update_function"] = None
configuration["node_assumption"] = False

N = 20
max_size = 4
H = 5
enforce_connectivity = True


def get_outputs(configuration, N, max_size, H, eps_vals=[0.0, 0.25, 0.5, 0.75, 1.0]):
    # with different epsilons
    outputs = dict()
    for epsilon2 in eps_vals:
        for epsilon3 in eps_vals:
            avg_nodes = 0
            dag_edges = []
            epsilons = {2: epsilon2, 3: epsilon3}
            for run in range(configuration["num_hypergraphs"]):
                hyperedges = random_nested_hypergraph(N, max_size, H, epsilons)
                if enforce_connectivity:
                    while not check_hyperedges_connectivity(hyperedges):
                        hyperedges = random_nested_hypergraph(N, max_size, H, epsilons)
                nodes = set([u for he in hyperedges for u in he])
                for node in nodes:
                    hyperedges.append((node,))
                dag, _, _ = encapsulation_dag(hyperedges)
                dag_edges.append(dag.number_of_edges())
                num_nodes_used = len(set([node for he in hyperedges for node in he]))
                avg_nodes += num_nodes_used
                #configuration["initial_active"] = len([he for he in hyperedges if len(he) == 2])
                configuration["initial_active"] = len(nodes)
                exp = run_many_simulations(hyperedges, configuration)
                exp["nodes_activated"] /= num_nodes_used
                num_edges = len(hyperedges)
                exp["edges_activated"] /= num_edges
                del exp["activated_edge_sizes"]
                if run == 0:
                    outputs[(epsilon2, epsilon3)] = exp
                else:
                    for key in exp:
                        outputs[(epsilon2, epsilon3)][key] = np.vstack((outputs[(epsilon2, epsilon3)][key], exp[key]))
            print(f"{(epsilon2, epsilon3)}, Avg nodes: {avg_nodes / configuration['num_hypergraphs']}, Avg DAG edges: {np.mean(dag_edges)}")
    return outputs


outputs_dict = dict()
configuration["seed_function"] = "uniform"
outputs_dict["Uniform"] = get_outputs(configuration, N, max_size, H)

configuration["seed_function"] = smallest_first_seed
outputs_dict["Smallest First"] = get_outputs(configuration, N, max_size, H)

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14,8), squeeze=False)
row_idx = 0
for seed_name in outputs_dict:
    outputs = outputs_dict[seed_name]
    # Get the subset of values we will show in the plots
    plot_dict = {(fr"$\epsilon_2={e2}, \epsilon_3={e3}$"):op
                 for (e2, e3), op in outputs.items() if e2 in [1.0, 0.0] and e3 in [1.0, 0.0]}

    # Compute heatmap of average time until maximum num of edges are activated
    eps_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    arr_max = np.zeros((len(eps_vals), len(eps_vals)))
    arr_val = np.zeros((len(eps_vals), len(eps_vals)))
    for arr_row, ep2 in enumerate(eps_vals):
        for col_idx, ep3 in enumerate(eps_vals):
            output_dict = outputs[ep2, ep3]
            arr_max[arr_row,col_idx] = output_dict["edges_activated"].cumsum(axis=1).argmax(axis=1).mean()
            arr_val[arr_row,col_idx] = output_dict["edges_activated"].cumsum(axis=1).max(axis=1).mean()


    x = list(range(configuration["steps"]+1))
    labels = ["% Nodes Activated", "% Edges Activated"]
    for col_idx, key in enumerate(["nodes_activated", "edges_activated"]):
        for exp in plot_dict:
            mean = np.mean(np.cumsum(plot_dict[exp][key], axis=1), axis=0)
            std = np.std(np.cumsum(plot_dict[exp][key], axis=1), axis=0)
            axs[row_idx][col_idx].plot(x, mean, label=exp)
            axs[row_idx][col_idx].fill_between(x, mean-std, mean+std, alpha=0.15)

        axs[row_idx][col_idx].set(xlabel="Time", ylabel=labels[col_idx], ylim=(0, 1.1))
        axs[row_idx][col_idx].spines['top'].set_visible(False)
        axs[row_idx][col_idx].spines['right'].set_visible(False)

    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncols=4, loc='upper left', bbox_to_anchor=(0.035, -0.001), frameon=False)

    col_idx += 1
    hm = axs[row_idx][col_idx].matshow(arr_val, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(hm, ax=axs[row_idx][col_idx], location='right', shrink=0.7)
    axs[row_idx][col_idx].set(xticks=list(range(len(eps_vals))), yticks=list(range(len(eps_vals))),
                              xticklabels=eps_vals, yticklabels=eps_vals,
                              xlabel=r"$\epsilon_3$", ylabel=r"$\epsilon_2$");
    axs[row_idx][col_idx].set_title("Mean Edges Activated (%)", pad=20, size=14)
    axs[row_idx][col_idx].xaxis.tick_bottom()

    row_idx += 1

fig.text(-0.001, 0.72, "Uniform\nSeeds", size=16, ha="center")
fig.text(-0.001, 0.25, "Smallest First\nSeeds", size=16, ha="center")
fig.subplots_adjust(wspace=0.4, hspace=0.5)
fig.savefig("../results/plots/RNHM-epsilon-comparison.pdf", bbox_inches='tight')
