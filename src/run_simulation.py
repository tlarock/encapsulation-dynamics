import sys
import os
import pickle
import argparse

from configparser import ConfigParser
from pathlib import Path

import xgi

import numpy as np
from utils import read_data, read_random_hyperedges
from update_rules import *
# ToDo: There is probably a nicer way to export this!
UPDATE_FUNCT_MAP = {
    "up": absolute_update_up,
    "down": absolute_update_down
}

from selection_rules import *
# ToDo: There is probably a nicer way to export this!
SELECTION_FUNCT_MAP = {
    "uniform": uniform_inactive,
    "biased": biased_inactive,
    "inverse": inverse_inactive
}

from simulation import *
from plot_simulation_results import *

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to configuration file.")
    parser.add_argument("config_key", type=str, help="Key in config file. Usualy dataset name.")
    parser.add_argument("selection_funct", type=str, help="Name of selection function to use.")
    parser.add_argument("update_funct", type=str, help="Name of update function to use.")
    parser.add_argument("ncpus", type=int, help="Number of CPUS to use.")
    parser.add_argument("--default_key", type=str, help="Default key for config file.", default="default-arc", required=False)
    parser.add_argument("--randomization_number", type=int, help="If >=0, will \
                        try to run on that randomization of the dataset.",
                        default=-1, required=False)
    args = parser.parse_args()
    config_file = args.config_file
    config_key = args.config_key
    selection_name = args.selection_funct
    update_name = args.update_funct
    ncpus = args.ncpus
    default_key = args.default_key
    randomization_number = args.randomization_number
    # Parse configuration file
    config = ConfigParser(os.environ)
    config.read(config_file)

    # Read dataset name and location
    dataset_name = config[config_key]["dataset_name"]
    data_prefix = config[default_key]["data_prefix"]

    results_prefix = config[default_key]["results_prefix"]
    results_path = f"{results_prefix}{dataset_name}/"

    # Create output directory
    Path(results_path).mkdir(parents=True, exist_ok=True)

    results_path += f"{dataset_name}"

    if randomization_number < 0:
        dataset_path = f"{data_prefix}{dataset_name}/{dataset_name}-"

        # Get the list of hyperedges from Austin's format
        hyperedges = read_data(dataset_path, multiedges=False)
    else:
        # Get the list of randomized hyperedges
        random_path = f"{data_prefix}{dataset_name}/"
        hyperedges = read_random_hyperedges(random_path + f"randomizations/random-simple-{randomization_number}.txt")
        results_path += f"_randomization-{randomization_number}"

    # Read configuration parameters
    configuration = {
        "initial_active": config[config_key].getint("initial_active"),
        "steps": config[config_key].getint("steps"),
        "active_threshold": config[config_key].getint("active_threshold"),
        "num_simulations": config[config_key].getint("num_simulations"),
        "single_edge_update": config[config_key].getboolean("single_edge_update"),
        "selection_function": SELECTION_FUNCT_MAP[selection_name],
        "update_function": UPDATE_FUNCT_MAP[update_name]
    }

    print(f"Running {selection_name} {update_name}")
    if ncpus > 1:
        output = run_many_parallel(hyperedges, configuration, ncpus)
    else:
        output = run_many_simulations(hyperedges, configuration)

    # Output data
    output_filename = results_path
    output_filename += f"_{selection_name}_{update_name}-{configuration['active_threshold']}"
    output_filename += f"_runs-{configuration['num_simulations']}"

    with open(output_filename + ".pickle", "wb") as fpickle:
        pickle.dump(output, fpickle)

    print("Done")
