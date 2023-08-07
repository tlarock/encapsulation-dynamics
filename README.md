# Encapsulation Dynamics
This repository contains code implementing the measurements and dynamical processes on hypergraphs described in:

* [LaRock, T. and Lambiotte, R. Encapsulation Structure and Dynamics in Hypergraphs. arXiv preprint. July 2023. arXiv:2307.04613.](https://arxiv.org/abs/2307.04613)

# Requirements
The code was written under python 3.9.15 and relies on standard libraries including `numpy`, `scipy`, `matplotlib`, and `networkx`. It also relies heavily on the `xgi`, the compleX Group Interactions package.

# Installation
This repository does not implement a package, therefore beyond installing the required packages, there is no installation.

# Usage
There are two main ways to run dynamical simulations. The file `run_simulations.py` can be used in conjunction with a configuration file to run simulations from the command line. Alternatively, simulation parameters can be specified directly in a dictionary and passed to one of the `run_*_simulations` functions found in `simulations.py`.

## Simulations from the command line using `run_simulations.py` and configuration file
To run a simulation from the command line, navigate to the `src` directory and check the output of `python run_simulations.py -h`. You will need to set up a configuration file, or simply use `configurations.ini` already included in the repository. The configuration file needs two sections, one to set default paths, and another for the simulation you want to run. Here is an example:

```
# Default paths to data and results directories
[default-local]
data_prefix = ../data/
results_prefix = ../results/

# Configuration for a simulation on the primary school contact data
[contact-primary-school]
dataset_name = contact-primary-school
initial_active = 1
steps = 25
active_threshold = 1
num_simulations = 25
read_function = read_data
```

Once you have this configuration file, you can run a simulation with:
``` 
python run_simulation.py configuration_file.ini contact-primary-school simultaneous encapsulation-immediate NCPU
```

Using the configuration file, this command will run 25 simulations of 25 steps using NCPUs and a threshold of 1 active subhyperedge. Seeding will be a single uniform random edge (the default combined with `initial_active=` in the configuraiton file). Both the number of seeds and the active threshold can also be controlled from the command line for convenience when running many simulations with different parameters. Results are then saved as serialized pickle files in the `../results/` directory.

## Simulations using API
You can also run simulations from within a python session by specifying a configuration dictionary directly. Here is an example:

```
from utils import read_data, largest_connected_component
from layer_randomization import layer_randomization
from seed_functions import smallest_first_seed
from simulation import run_many_simulations
from plot_simulation_results import plot_cumulative_averages

dataset_name = "email-Enron"
dataset = f"../data/{dataset_name}/{dataset_name}-"
hyperedges = read_data(dataset, multiedges=False)
hyperedges = largest_connected_component(hyperedges, remove_single_nodes=True)
configuration = { 
    "seeding_strategy": "edge",
    "seed_function": smallest_first_seed,
    "initial_active": 10,
    "num_simulations": 10,
    "steps": 10,
    "active_threshold": 1,
    "selection_name": "simultaneous",
    "selection_function": None,
    "update_name": "encapsulation-immediate",
    "update_function": None,
    "encapsulation_update": True,
    "node_assumption": False
}   
output_observed = run_many_simulations(hyperedges, configuration)
random_hyperedges = largest_connected_component(layer_randomization(hyperedges), remove_single_nodes=True)
output_random = run_many_simulations(random_hyperedges, configuration)
fig, axs = plot_cumulative_averages(output_observed, output_random, normalized=False)
```

## Constructing encapsulation and overlap structures
The file `encapsulation_dag.py` implements separate computation of the encapsulation and overlap line graphs. You can also find implementations of these functions in the [xgi](https://github.com/xgi-org/xgi) package, specifically in the [convert module (permalink to merge commit)](https://github.com/xgi-org/xgi/tree/ab2a2c7ddb9ef32f26ea216171c9715e49712f9b/xgi/convert).

*Note:* This README is a work in progress. If you have any questions, open an issue or email me at [larock@maths.ox.ac.uk](mailto:larock@maths.ox.ac.uk).
