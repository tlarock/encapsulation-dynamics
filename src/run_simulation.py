import sys
import os
import pickle
import argparse

from configparser import ConfigParser
from pathlib import Path

import xgi

import numpy as np
from utils import read_data, read_hyperedges, largest_connected_component, drop_hyperedges_by_size, check_hyperedges_connectivity
from layer_randomization import layer_randomization

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
    "inverse": inverse_inactive,
    "simultaneous": None
}

from seed_functions import *
SEED_FUNCT_MAP = {
    "biased_seed": biased_seed,
    "inverse_biased_seed": inverse_biased_seed,
    "uniform": "uniform",
    "twonode": twonode_seed,
    "degree_biased": degree_biased_seed,
    "inverse_degree": inverse_degree_biased,
    "size_biased": size_biased_seed,
    "inverse_size": inverse_size_biased
}

from simulation import *
from plot_simulation_results import *


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to configuration file.")
    parser.add_argument("config_key", type=str, help="Key in config file. Usualy dataset name.")
    parser.add_argument("selection_funct", type=str, help="Name of selection function to use.")
    parser.add_argument("update_funct", type=str, help="Name of update function to use.")
    parser.add_argument("ncpus", type=int, help="Number of CPUS to use.")
    parser.add_argument("--seeding_strategy", type=str, required=False, choices=["node", "edge"], default="node",
                        help="Seeding strategy to use. Default is node seeding.")
    parser.add_argument("--default_key", type=str, help="Default key for config file.", default="default-arc", required=False)
    parser.add_argument("--randomization_number", type=int, help="If >=0, will \
                        try to run on that randomization of the dataset.",
                        default=-1, required=False)
    parser.add_argument("--num_seeds_override", type=int, help="If >=1, overrides \
                        number of seeds argument in config file.",
                        default=-1, required=False)
    parser.add_argument("--threshold_override", type=int, help="If >=0, overrides \
                        threshold argument in config file.",
                        default = -1, required=False)
    parser.add_argument("--largest_cc", action="store_true",
                        help="If given, compute the largest connected component \
                        of the hypergraph before running simulations.")
    parser.add_argument("--seed_funct", required=False, default="uniform",
                        choices=set(SEED_FUNCT_MAP.keys()),
                        help="Name of seed function.")
    parser.add_argument("--drop_hyperedges_size", required=False, default=-1, type=int,
                        help="Given a size, drop all hyperedges of that size.")
    parser.add_argument("--layer_randomization", required=False,
                        action="store_true", help="If given, apply layer randomization to the dataset.")

    args = parser.parse_args()
    return args


def read_input(config, config_key, data_prefix, dataset_name,
              randomization_number, results_path, _layer_randomization):
    if config[config_key]["read_function"] == "read_hyperedges" and randomization_number < 0:
        dataset_path = f"{data_prefix}{dataset_name}/{dataset_name}.txt"
        hyperedges = read_hyperedges(dataset_path)
    elif randomization_number >= 0:
        # Get the list of randomized hyperedges
        random_path = f"{data_prefix}{dataset_name}/"
        hyperedges = read_hyperedges(random_path + f"randomizations/random-simple-{randomization_number}.txt")
        results_path += f"_randomization-{randomization_number}"
    elif config[config_key]["read_function"] == "read_data":
        # Get the list of hyperedges from Austin's format
        dataset_path = f"{data_prefix}{dataset_name}/{dataset_name}-"
        hyperedges = read_data(dataset_path, multiedges=False)

    if _layer_randomization:
        print("Applying layer randomization.")
        hyperedges = layer_randomization(hyperedges)
        results_path += f"_layer_randomization"

    return hyperedges, results_path

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_command_line()
    config_file = args.config_file
    config_key = args.config_key
    selection_name = args.selection_funct
    update_name = args.update_funct
    ncpus = args.ncpus
    seeding_strategy = args.seeding_strategy
    default_key = args.default_key
    randomization_number = args.randomization_number
    num_seeds_override = args.num_seeds_override
    threshold_override = args.threshold_override
    largest_cc = args.largest_cc
    seed_funct = args.seed_funct
    drop_size = args.drop_hyperedges_size
    _layer_randomization = args.layer_randomization

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

    hyperedges, results_path = read_input(config, config_key, data_prefix,
                                         dataset_name, randomization_number,
                                         results_path, _layer_randomization)
    if drop_size > 0:
        print(f"Dropping hyperedges of size {drop_size}")
        hyperedges = drop_hyperedges_by_size(hyperedges, drop_size)

    if largest_cc:
        if not check_hyperedges_connectivity(hyperedges):
            print("Computing largest connected component.")
            hyperedges = largest_connected_component(hyperedges, remove_single_nodes=True)
        else:
            print("Hyperedges already connected. Not computing largest component.")

    # Check for override parameters passed via the command line
    if num_seeds_override >= 1:
        initial_active = num_seeds_override
    else:
        initial_active = config[config_key].getint("initial_active")

    if threshold_override >= 0:
        active_threshold = threshold_override
    else:
        active_threshold = config[config_key].getint("active_threshold")

    # Read configuration parameters
    configuration = {
        "initial_active": initial_active,
        "steps": config[config_key].getint("steps"),
        "active_threshold": active_threshold,
        "num_simulations": config[config_key].getint("num_simulations"),
        #"single_edge_update": config[config_key].getboolean("single_edge_update"),
        "selection_name": selection_name,
        "selection_function": SELECTION_FUNCT_MAP[selection_name],
        "update_name": update_name,
        "update_function": UPDATE_FUNCT_MAP[update_name],
        "seeding_strategy": seeding_strategy,
        "seed_function": SEED_FUNCT_MAP[seed_funct]
    }

    print(f"Running simulation with the following parameters:\
            \nHyperedge Selection: {selection_name}\nUpdate Rule: {update_name}\
          \nSeed strategy: {seeding_strategy}\nSeed Function: {seed_funct}\nNumber of Seeds:\
          {initial_active}\nThreshold: {configuration['active_threshold']}")

    if ncpus > 1:
        output = run_many_parallel(hyperedges, configuration, ncpus)
    else:
        output = run_many_simulations(hyperedges, configuration)

    output["total_edges"] = len(hyperedges)
    output["total_nodes"] = len(set([u for he in hyperedges for u in he]))
    # Output data
    output_filename = results_path
    output_filename += f"_{selection_name}_{update_name}_"
    output_filename += f"steps-{configuration['steps']}_"
    output_filename += f"t-{configuration['active_threshold']}_"
    output_filename += f"ia-{configuration['initial_active']}_"
    output_filename += f"runs-{configuration['num_simulations']}_"
    output_filename += seeding_strategy

    # ToDo: This naming might be more complicated than necessary?
    if seed_funct == "biased_seed":
        output_filename += "_biased"
    elif seed_funct == "inverse_biased_seed":
        output_filename += "_inverse-biased"
    elif seed_funct == "twonode":
        output_filename += "_twonode"
    elif "deg" in seed_funct or "size" in seed_funct:
        output_filename += "_" + seed_funct

    if drop_size > 0:
        output_filename += f"_drop_size-{drop_size}"

    with open(output_filename + ".pickle", "wb") as fpickle:
        pickle.dump(output, fpickle)

    print("Done")
