import numpy as np
from collections import defaultdict
from itertools import combinations_with_replacement

import xgi
from utils import check_hyperedges_connectivity

from update_rules import *
from selection_rules import *
from simulation import *
from seed_functions import *
from plot_simulation_results import *
from encapsulation_dag import encapsulation_dag
from random_nested_hypergraph import random_nested_hypergraph

import matplotlib.pyplot as plt
import matplotlib_defaults
from matplotlib.gridspec import GridSpec
from matplotlib import ticker


# Simulation parameters
configuration = {
    "initial_active": 1,
    "steps": 10,
    "active_threshold": 1,
    "num_simulations": 5
}

configuration["num_hypergraphs"] = 5
configuration["seeding_strategy"] = "edge"
configuration["selection_name"] = "simultaneous"
configuration["selection_function"] = None
configuration["encapsulation_update"] = True
configuration["update_name"] = "encapsulation-immediate"
configuration["update_function"] = None
configuration["node_assumption"] = False

N = 20
max_size = 4
H = 5
enforce_connectivity = True

seed_values = np.array(list(range(1, 40)))
epsilon_vals = [0.0, 0.5, 1.0]

seed_functs = ["uniform", "smallest_first"]

epsilon_combinations = list(combinations_with_replacement(epsilon_vals, 2))
epsilon_combinations.append(epsilon_combinations[-1])
epsilon_combinations[-2] = (1.0, 0.5)
act_results = {seed_funct: {eps: {ia: []
                              for ia in seed_values
                             }
                     for eps in epsilon_combinations}
            for seed_funct in seed_functs}
# Need to accumulate means and standard deviations across more simulations
num_edges_dict = {eps:0.0 for eps in epsilon_combinations}
dag_edges_dict = {eps:0.0 for eps in epsilon_combinations}
for eps in epsilon_combinations:
    print(eps)
    epsilon_2, epsilon_3 = eps
    epsilons = {2:epsilon_2, 3:epsilon_3}
    for sample in range(configuration["num_hypergraphs"]):
        hyperedges = random_nested_hypergraph(N, max_size, H, epsilons)
        if enforce_connectivity:
            while not check_hyperedges_connectivity(hyperedges):
                hyperedges = random_nested_hypergraph(N, max_size, H, epsilons)
        nodes = set([u for he in hyperedges for u in he])
        for node in nodes:
            hyperedges.append((node,))
        num_edges = len(hyperedges)
        num_edges_dict[eps] += num_edges
        dag, _, _ = encapsulation_dag(hyperedges)
        dag_edges_dict[eps] += dag.number_of_edges()
        for seed_funct in seed_functs:
            if seed_funct == "smallest_first":
                configuration["seed_function"] = smallest_first_seed
            else:
                configuration["seed_function"] = seed_funct

            for initial_active in seed_values:
                configuration["initial_active"] = initial_active
                exp = run_many_simulations(hyperedges, configuration)
                exp["edges_activated"] /= num_edges
                act_results[seed_funct][eps][initial_active] += exp["edges_activated"].cumsum(axis=1).max(axis=1).tolist()

    num_edges_dict[eps] /= configuration["num_hypergraphs"]
    dag_edges_dict[eps] /= configuration["num_hypergraphs"]

act_mean = dict()
act_std = dict()
for seed_funct in act_results:
    act_mean[seed_funct] = dict()
    act_std[seed_funct] = dict()
    for eps in act_results[seed_funct]:
        act_mean[seed_funct][eps] = []
        act_std[seed_funct][eps] = []
        for initial_active in seed_values:
            act_mean[seed_funct][eps].append(np.mean(act_results[seed_funct][eps][initial_active]))
            act_std[seed_funct][eps].append(np.std(act_results[seed_funct][eps][initial_active]))

fig, axs = plt.subplots(1, 3, figsize=(14, 4), squeeze=False, gridspec_kw={'width_ratios':[0.9, 0.6, 0.9]})
row_idx = 0
col_idx = 0
for seed_funct in seed_functs:
    denom = max(num_edges_dict.values())
    for eps in epsilon_combinations:
        label = fr"$\epsilon_2={eps[0]},\epsilon_3={eps[1]}$ ({dag_edges_dict[eps]} DAG Edges)"
        axs[row_idx][col_idx].errorbar(seed_values / denom, act_mean[seed_funct][eps], act_std[seed_funct][eps], alpha=0.7, label=label)
    axs[row_idx][col_idx].set(xlabel="Fraction of Seed Edges", ylabel="Avg. Proportion Edges Activated", title=seed_funct,
                             #xticks = seed_values / denom,
                              ylim=(-0.1, 1.1)
                             )

    axs[row_idx][col_idx].plot(seed_values / denom, seed_values / denom, ls='--', alpha=0.3, color="black")
    positions = (seed_values / denom)[::2]
    labels = np.round(positions, 2)
    axs[row_idx][col_idx].xaxis.set_major_locator(ticker.FixedLocator(positions))
    axs[row_idx][col_idx].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    axs[row_idx][col_idx].set_xticklabels(labels, rotation=45)
    axs[row_idx][col_idx].spines['top'].set_visible(False)
    axs[row_idx][col_idx].spines['right'].set_visible(False)
    col_idx = 2

h, l = axs[row_idx][col_idx].get_legend_handles_labels()
axs[0][1].axis('off')
fig.legend(reversed(h), reversed(l), ncols=1, loc='lower left', bbox_to_anchor=(0.365, 0.25), frameon=False)

fig.tight_layout()
fig.savefig(f"../results/plots/RNHM-synthetic-seed-simulation-unif-vs-smallest.pdf", dpi=150)
