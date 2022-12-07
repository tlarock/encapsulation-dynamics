import numpy as np
import matplotlib.pyplot as plt


def plot_cumulative(configuration, results_obs, results_rnd):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), squeeze=False)
    fig.subplots_adjust(wspace=0.3)

    x = list(range(configuration["steps"]))
    axs[0][0].plot(x, np.cumsum(results_obs["nodes_activated"]), label="Observed")
    axs[0][0].plot(x, np.cumsum(results_rnd["nodes_activated"]), label="Randomized")
    axs[0][0].set(xlabel="Time", ylabel="Cumulative Nodes Activated")
    axs[0][0].legend()

    axs[0][1].plot(x, np.cumsum(results_obs["edges_activated"]), label="Observed")
    axs[0][1].plot(x, np.cumsum(results_rnd["edges_activated"]), label="Randomized")
    axs[0][1].set(xlabel="Time", ylabel="Cumulative Edges Activated")
    return fig, axs

def plot_cumulative_averages(configuration, output_obs, output_rnd):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), squeeze=False)
    fig.subplots_adjust(wspace=0.3)

    x = list(range(configuration["steps"]))
    labels = ["Cumulative Nodes Activated", "Cumulative Edges Activated"]
    for col_idx, key in enumerate(["nodes_activated", "edges_activated"]):
        mean = np.mean(np.cumsum(output_obs[key], axis=1), axis=0)
        std = np.std(np.cumsum(output_obs[key], axis=1), axis=0)
        axs[0][col_idx].plot(x, mean, label="Observed")
        axs[0][col_idx].fill_between(x, mean-std, mean+std, alpha=0.3)

        mean = np.mean(np.cumsum(output_rnd[key], axis=1), axis=0)
        std = np.std(np.cumsum(output_rnd[key], axis=1), axis=0)
        axs[0][col_idx].plot(x, mean, label="Randomized")
        axs[0][col_idx].fill_between(x, mean-std, mean+std, alpha=0.3)
        axs[0][col_idx].set(xlabel="Time", ylabel=labels[col_idx])
        axs[0][col_idx].legend()
    return fig, axs
