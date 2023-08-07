from plot_simulation_results import *
from utils import read_pickles

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib_defaults

title_mapping = {"subface": "Non-Strict Encapsulation",
                 "subface-strict": "Strict Encapsulation",
                 "encapsulation-immediate-strict": "Strict Encapsulation",
                  "encapsulation-immediate": "Non-strict Encapsulation",
                "encapsulation-empirical-strict": "Empirical Encapsulation"}
filename_mapping = {"subface": "encapsulation", "subface-strict": "encapsulation-strict",  "encapsulation-all": "encapsulation-all"}

results_path = "../results/"
params_dict = {
    "seeding_strategy": "edge",
    "selection": "simultaneous",
    "layer_randomization": True
}
key = "edges_activated"
step = 25


datasets = [
    ("coauth-MAG-Geology", 50, 50, 500_000),
    ("coauth-MAG-History", 50, 50, 100_000),
    ("contact-high-school", 25, 25, 5000),
    ("contact-primary-school", 25, 25, 10000),
    ("email-Enron", 50, 50, 1000),
    ("email-Eu", 50, 50, 20000)
]

params_dict["threshold"] = "all"
normalize = True
seed_vals = np.array([1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000, 500000])

for thresh in [1, "all"]:
    params_dict["threshold"] = thresh
    for update in ["encapsulation-immediate", "encapsulation-empirical", "encapsulation-immediate-strict", "encapsulation-empirical-strict"]:
    #for update in ["encapsulation-empirical-strict"]:
        params_dict["update"] = update

        fig = plt.figure(figsize=(20,6))
        gridsize = (2, len(datasets))

        col_idx = 0
        for dataset_name, steps, runs, max_seed in datasets:
            results_prefix = f"{results_path}{dataset_name}/{dataset_name}"
            params_dict["steps"] = steps
            params_dict["runs"] = runs
            curr_seed_vals = seed_vals[seed_vals <= max_seed]
            if dataset_name == "coauth-DBLP":
                curr_seed_vals = curr_seed_vals[curr_seed_vals != 500]

            ax1 = plt.subplot2grid(gridsize, (0, col_idx))
            ax2 = plt.subplot2grid(gridsize, (1, col_idx))
            labels = []
            for biased_seed in ["uniform", "size_biased", "inverse_size", "smallest_first"]:
                params_dict["seed_funct"] = biased_seed
                activations_data = {
                    "obs": [],
                    "obs_std":[],
                    "rnd": [],
                    "rnd_std":[],
                    "obs_nonorm": [],
                    "rnd_nonorm": []
                }
                for num_seeds in curr_seed_vals:
                    params_dict["ia"] = num_seeds
                    output_obs, output_rnd = read_pickles(results_prefix, params_dict=params_dict)
                    obs_norm = rnd_norm = 1
                    obs = output_obs[key].cumsum(axis=1)[:, step]
                    rnd = output_rnd[key].cumsum(axis=1)[:, step]
                    obs -= num_seeds
                    rnd -= num_seeds
                    activations_data["obs_nonorm"].append(obs.mean())
                    activations_data["rnd_nonorm"].append(rnd.mean())
                    obs_norm = output_obs["total_edges"] - num_seeds
                    rnd_norm = output_rnd["total_edges"] - num_seeds

                    if not normalize:
                        activations_data["obs"].append((obs).mean())
                        activations_data["obs_std"].append((obs).std())

                        activations_data["rnd"].append((rnd).mean())
                        activations_data["rnd_std"].append((rnd).std())
                    else:
                        activations_data["obs"].append((obs / obs_norm).mean())
                        activations_data["obs_std"].append((obs / obs_norm).std())

                        activations_data["rnd"].append((rnd / rnd_norm).mean())
                        activations_data["rnd_std"].append((rnd / rnd_norm).std())



                ax1.set_title(dataset_name, fontsize=14)
                v = ax1.errorbar(curr_seed_vals, activations_data["obs"], yerr=activations_data["obs_std"],
                                 marker="o",
                                 alpha=0.7,
                                 label=biased_seed)
                #add_label(v, biased_seed, labels)
                ax1.errorbar(curr_seed_vals, activations_data["rnd"], yerr=activations_data["rnd_std"],
                             marker="^",
                             linestyle='--',
                             alpha=0.5,
                             color=v[0].get_markerfacecolor())

                ax2.plot(curr_seed_vals, np.array(activations_data["obs"]) - np.array(activations_data["rnd"]),
                                 marker="o",
                                 alpha=0.7,
                                 label=biased_seed)

            if col_idx == 0:
                ax1.legend(ncols=4, bbox_to_anchor=(2.75,-1.8), loc="upper left", frameon=False)

            if col_idx == 0:
                ax1.set_ylabel("Edges Activated")
                ax2.set_ylabel(r"Observed - Random")

            #if norm_type != "none":
            #    ax1.set_ylim((-0.02, 1.02))

            ax2.hlines(0.0, 1, max(curr_seed_vals),  linestyles='--', alpha=0.5, color="black")
            #ax2.set_ylim(0.5, 7)
            xticks = [10**i for i in range(0, int(max(np.log10(curr_seed_vals))+1))]
            #if len(xticks) > 3:
            #    xticks = [10**i for i in range(0, int(max(np.log10(curr_seed_vals))+1), 2)]

            ax1.set_ylim(-0.1, 1.1)
            ax1.hlines(1.0, 1, max(curr_seed_vals), color="black", alpha=0.3, linestyle="--")
            ax2.set_ylim(-0.1, 0.7)
            for ax in [ax1, ax2]:
                ax.set(xscale="log", xticks=xticks)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            #y = curr_seed_vals / output_obs["total_edges"]
            #ax1.plot(curr_seed_vals, y, marker='o', alpha=0.5, color="black")
            col_idx += 1

        fig.subplots_adjust(wspace=0.5, hspace=0.2)
        title = update
        if update in title_mapping:
            title = title_mapping[update]

        fig.supxlabel("# Seeds", size=15, y=-0.05)

        filename = update
        if update in filename_mapping:
            filename = filename_mapping[update]

        if update != "down":
            fig.suptitle(fr"{title} Dynamics, {step} steps, $\tau=${params_dict['threshold']}", y=1)
            fig.savefig(f"../results/plots/simultaneous-seed-simulations/seed_simulation_{filename}_{params_dict['threshold']}_combo.pdf", bbox_inches="tight")
        else:
            fig.suptitle(fr"Threshold Dynamics, {step} steps, $\tau={params_dict['threshold']}$", y=1)
            fig.savefig(f"../results/plots/simultaneous-seed-simulations/seed_simulation_{filename}_{params_dict['threshold']}_combo.pdf", bbox_inches="tight")
