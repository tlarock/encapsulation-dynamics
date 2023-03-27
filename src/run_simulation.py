import sys
import os
import pickle

from configparser import ConfigParser
from pathlib import Path

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

    fig, axs = plot_cumulative_averages_sizes(configuration, output_obs, output_rnd)
    plot_filename = results_path
    plot_filename += f"_{selection_name}_{update_name}-{configuration['active_threshold']}"
    plot_filename += f"_runs-{configuration['num_simulations']}"
    fig.tight_layout()
    fig.savefig(plot_filename + ".pdf", dpi=150)
    fig.savefig(plot_filename + ".png", dpi=150)

    # Output data
    with open(plot_filename + ".pickle", "wb") as fpickle:
        pickle.dump({"observed":output_obs, "random":output_rnd}, fpickle)

    return None


def main_funct(hyperedges, random_hyperedges, configuration, results_path):
    args = []
    for selection_name, selection_funct in [("uniform", uniform_inactive),
                                            ("biased", biased_inactive),
                                            ("inverse", inverse_inactive)]:
        for update_name, update_funct in [("up", absolute_update_up),
                                          ("down", absolute_update_down)]:
            args.append((hyperedges, random_hyperedges, configuration, selection_name, selection_funct,
                     update_name, update_funct, results_path))

    with Pool(len(args)) as p:
        p.starmap(run_and_plot, args)
    return None

if __name__ == "__main__":
    config_file = sys.argv[1]
    config_key = sys.argv[2]
    config = ConfigParser(os.environ)
    config.read(config_file)

    # Read dataset name and location
    dataset_name = config[config_key]["dataset_name"]
    data_prefix = config["default"]["data_prefix"]

    dataset_path = f"{data_prefix}{dataset_name}/{dataset_name}-"

    # Get the list of hyperedges from Austin's format
    hyperedges = read_data(dataset_path, multiedges=False)

    # Get the list of randomized hyperedges
    random_path = f"{data_prefix}{dataset_name}/"
    random_hyperedges = read_random_hyperedges(random_path + "randomizations/random-simple-0.txt")

    results_prefix = config["default"]["results_prefix"]
    results_path = f"{results_prefix}{dataset_name}/"

    # Create output directory
    Path(results_path).mkdir(parents=True, exist_ok=True)

    results_path += f"{dataset_name}"

    # Read configuration parameters
    configuration = {
        "initial_active": config[config_key].getint("initial_active"),
        "steps": config[config_key].getint("steps"),
        "active_threshold": config[config_key].getint("active_threshold"),
        "num_simulations": config[config_key].getint("num_simulations"),
        "single_edge_update": config[config_key].getboolean("single_edge_update")
    }

    main_funct(hyperedges, random_hyperedges, configuration, results_path)
    print("Done")
