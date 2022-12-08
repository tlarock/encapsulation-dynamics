import xgi
import numpy as np
from multiprocessing import Pool
from utils import read_data, read_random_hyperedges
from update_rules import *
from selection_rules import *
from simulation import *
from plot_simulation_results import *

def run_and_plot(hyperedges, random_hyperedges, configuration, selection_name, selection_funct,
                 update_name, update_funct, results_path):
    print(f"Running {selection_name} {update_name}")
    configuration["selection_function"] = selection_funct
    configuration["update_function"] = update_funct

    output_obs = run_many_simulations(hyperedges, configuration)
    output_rnd = run_many_simulations(random_hyperedges, configuration)
    fig, axs = plot_cumulative_averages(configuration, output_obs, output_rnd)
    plot_filename = results_path
    plot_filename += f"_{selection_name}_{update_name}-{configuration['active_threshold']}"
    plot_filename += f"_runs-{configuration['num_simulations']}"
    fig.tight_layout()
    fig.savefig(plot_filename + ".pdf", dpi=150)
    fig.savefig(plot_filename + ".png", dpi=150)
    return None


def main_funct(hyperedges, random_hyperedges, configuration, results_path):
    args = []
    for selection_name, selection_funct in [("uniform", uniform_inactive), ("biased", biased_inactive)]:
        for update_name, update_funct in [("up", absolute_update_up), ("down", absolute_update_down)]:
            args.append((hyperedges, random_hyperedges, configuration, selection_name, selection_funct,
                     update_name, update_funct, results_path))

    with Pool(len(args)) as p:
        p.starmap(run_and_plot, args)
    return None

if __name__ == "__main__":
    # Get the list of hyperedges from Austin's format
    #dataset_name = "contact-high-school"
    #dataset_name = "contact-primary-school"
    dataset_name = "coauth-MAG-Geology-full"

    dataset_path = f"../data/{dataset_name}/{dataset_name}-"
    hyperedges = read_data(dataset_path, multiedges=False)

    # Get the list of randomized hyperedges
    random_path = f"../data/{dataset_name}/"
    random_hyperedges = read_random_hyperedges(random_path + "randomizations/random-simple-0.txt")

    results_path = f"../results/{dataset_name}/{dataset_name}"


    configuration = {
        "initial_active": 15_000,
        "steps": 25000,
        "active_threshold": 1,
        "num_simulations": 5
    }

    #configuration = {
    #    "initial_active": 1,
    #    "steps": 5000,
    #    "active_threshold": 1,
    #    "num_simulations": 100
    #}

    main_funct(hyperedges, random_hyperedges, configuration, results_path)
