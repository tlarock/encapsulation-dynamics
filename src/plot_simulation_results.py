import numpy as np
import matplotlib.pyplot as plt

"""
    Plot a single result with time on the x-axis and cumulative
    number of nodes/edges activated on the y-axis.

    results_obs and results_rnd are outputs of run_simulation function
    in run_simulation.py.
"""
def plot_cumulative(configuration, results_obs, results_rnd,
                    first_label="Observed", second_label="Randomized"):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), squeeze=False)
    fig.subplots_adjust(wspace=0.3)

    x = list(range(configuration["steps"]+1))
    axs[0][0].plot(x, np.cumsum(results_obs["nodes_activated"]),
                   label=first_label)
    axs[0][0].plot(x, np.cumsum(results_rnd["nodes_activated"]),
                   label=second_label)
    axs[0][0].set(xlabel="Time", ylabel="Cumulative Nodes Activated")
    axs[0][0].legend()

    axs[0][1].plot(x, np.cumsum(results_obs["edges_activated"]),
                   label=first_label)
    axs[0][1].plot(x, np.cumsum(results_rnd["edges_activated"]),
                   label=second_label)
    axs[0][1].set(xlabel="Time", ylabel="Cumulative Edges Activated")
    return fig, axs

"""
    Plot average cumulative activations of nodes/edges over multiple
    simulations.

    output_obs and output_rnd are outputs of run_many_simulations function
    in run_simulations.py.
"""
def plot_cumulative_averages(configuration, output_obs, output_rnd,
                    first_label="Observed", second_label="Randomized",
                             yscale="linear"):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), squeeze=False)
    fig.subplots_adjust(wspace=0.3)

    x = list(range(configuration["steps"]+1))
    labels = ["Cumulative Nodes Activated", "Cumulative Edges Activated"]
    for col_idx, key in enumerate(["nodes_activated", "edges_activated"]):
        mean = np.mean(np.cumsum(output_obs[key], axis=1), axis=0)
        std = np.std(np.cumsum(output_obs[key], axis=1), axis=0)
        axs[0][col_idx].plot(x, mean, label=first_label)
        axs[0][col_idx].fill_between(x, mean-std, mean+std, alpha=0.3)

        mean = np.mean(np.cumsum(output_rnd[key], axis=1), axis=0)
        std = np.std(np.cumsum(output_rnd[key], axis=1), axis=0)
        axs[0][col_idx].plot(x, mean, label=second_label)
        axs[0][col_idx].fill_between(x, mean-std, mean+std, alpha=0.3)
        axs[0][col_idx].set(xlabel="Time", ylabel=labels[col_idx],
                            yscale=yscale)
        axs[0][col_idx].legend()
    return fig, axs


"""
    Plot average cumulative activations of nodes/edges over multiple
    simulations, as well as the average size of activated hyperedges
    over time.

    output_obs and output_rnd are outputs of run_many_simulations function
    in run_simulations.py.
"""
def plot_cumulative_averages_sizes(configuration, output_obs, output_rnd,
                                   figsize=(10,3)):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, squeeze=False)
    fig.subplots_adjust(wspace=0.3)

    x = list(range(configuration["steps"]+1))
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


    # Plot the average size of the hyperedges
    # ToDo: I should plot real average as reference
    col_idx = 2
    key = "activated_edge_sizes"
    obs_cmean = np.zeros(output_obs[key].shape)
    obs_cmean[:, 0] = output_obs[key][:, 0]
    rnd_cmean = np.zeros(output_obs[key].shape)
    rnd_cmean[:, 0] = output_rnd[key][:, 0]
    for t in range(1, obs_cmean.shape[1]):
        obs_cmean[:, t] = np.nanmean(output_obs[key][:,0:t],
                                    where=output_obs[key][:,0:t]>0,
                                    axis=1)
        rnd_cmean[:, t] = np.nanmean(output_rnd[key][:,0:t],
                                    where=output_rnd[key][:,0:t]>0,
                                    axis=1)

    mean = np.mean(obs_cmean, axis=0)
    std = np.std(obs_cmean, axis=0)
    axs[0][col_idx].plot(x, mean, label="Observed")
    axs[0][col_idx].fill_between(x, mean-std, mean+std, alpha=0.3)

    mean = np.mean(rnd_cmean, axis=0)
    std = np.std(rnd_cmean, axis=0)
    axs[0][col_idx].plot(x, mean, label="Randomized")
    axs[0][col_idx].fill_between(x, mean-std, mean+std, alpha=0.3)
    axs[0][col_idx].set(xlabel="Time", ylabel="Average Size")

    return fig, axs


"""
    Plot average cumulative activations of nodes/edges over multiple
    simulations based on a dictionary with many simulation results.
"""
def plot_cumulative_fromdict(configuration, outputs):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), squeeze=False)
    fig.subplots_adjust(wspace=0.3)

    x = list(range(configuration["steps"]+1))
    labels = ["Cumulative Nodes Activated", "Cumulative Edges Activated"]
    for col_idx, key in enumerate(["nodes_activated", "edges_activated"]):
        for exp in outputs:
            mean = np.mean(np.cumsum(outputs[exp][key], axis=1), axis=0)
            std = np.std(np.cumsum(outputs[exp][key], axis=1), axis=0)
            axs[0][col_idx].plot(x, mean, label=exp)
            axs[0][col_idx].fill_between(x, mean-std, mean+std, alpha=0.15)

        axs[0][col_idx].set(xlabel="Time", ylabel=labels[col_idx])
    axs[0][0].legend()
    return fig, axs
