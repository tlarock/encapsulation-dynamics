# Encapsulation Dynamics
This repository contains code implementing the measurements and dynamical processes on hypergraphs described in:

* [LaRock, T. and Lambiotte, R. Encapsulation Structure and Dynamics in Hypergraphs. arXiv preprint. July 2023. arXiv:2307.04613.](https://arxiv.org/abs/2307.04613)

# Requirements
The code was written under python 3.9.15 and relies on standard libraries including `numpy`, `scipy`, `matplotlib`, and `networkx`. It also relies heavily on `xgi`, the compleX Group Interactions package.

# Installation
This repository does not implement a package, therefore beyond installing the requirements, there is no installation.

# Usage
There are two main ways to run simulations of encapsulation dynamics. The file `run_simulations.py` can be used in conjunction with a configuration file to run simulations from the command line. Alternatively, simulation parameters can be specified directly in a dictionary and passed to one of the `run_*_simulations` functions found in `simulations.py`.

## Simulations from the command line using `run_simulations.py` and configuration file
To run a simulation from the command line, first navigate to the `src` directory and check the output of `python run_simulations.py -h`, which will list the required and optional arguments to the simulation code.

You will need to set up a configuration file, or simply use `configurations.ini` already included in the repository. The configuration file will be parsed using the [configparser](https://docs.python.org/3/library/configparser.html) module and needs two sections, one to set default paths, and another for the simulation you want to run. Here is an example:

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

Once you have created this configuration file, you can run the simulations with:
``` 
python run_simulation.py configuration_file.ini contact-primary-school simultaneous encapsulation-immediate NCPU
```

Using the configuration file, this command will run 25 simulations of 25 steps using NCPUs (an integer number of CPUs to use) and a threshold of 1 active subhyperedge. Seeding will be a single uniform random edge (the default combined with `initial_active=1` in the configuraiton file; use command line option `--seed_funct` to control the seed function).

Both the number of seeds and the active threshold can also be controlled from the command line for convenience when running many simulations with different parameters. When given, command line inputs take precedence over configuration file parameters.

Results are saved as serialized [pickle](https://docs.python.org/3/library/pickle.html) files in the `../results/` directory.

## Simulations using the API directly
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

If you want to run simulations in parallel, you can swap `run_many_simulations(hyperedges, configuration)` for `run_many_parallel(hyperedges, configuration, ncpus)`, where `ncpus` is an `int` corresponding to the number of cpus to use.

## Important parameters
The relevant options for dynamics, controlled by `update_funct` positional argument (command line) and `configuration['update_name']` dictionary entry (API), are:
* `'encapsulation'`: encapsulation dynamics including all encapsulation relationships
* `'encapsulation-immediate'`: encapsulation dynamics including only immediate encapsulation relationships (i.e., `k->k-1` DAG edges)
* `'encapsulation-empirical'`: relaxation of `encapsulation-immediate` dynamics including relationships between size `k` and subhyperedges of maximum size `k'<k` existing in the hypergraph (e.g., a hyperedge of size 5 has no encapsulation relationships with hyperedges of size 4, but some with sizes 3 and 2, the edges to the size 3 hyperedges will "count" for the dynamics)

Dynamics are strict by default. For non-strict dynamics, specify `--node_assumption` or set `configuration['node_assumption']=True`.

To use the "all encapsulated hyperedges" threshold, specify `--encapsulation_all_thresh` or set `configuration['active_threshold']='all'`.

## Constructing encapsulation and overlap structures
The file `encapsulation_dag.py` implements separate computation of the encapsulation and overlap line graphs. You can also find implementations of these functions in the [xgi](https://github.com/xgi-org/xgi) package, specifically in the [convert module (permalink to merge commit)](https://github.com/xgi-org/xgi/tree/ab2a2c7ddb9ef32f26ea216171c9715e49712f9b/xgi/convert).

# Datasets
The datasets used in the paper were made available by Austin Benson and can be found on his website: [https://www.cs.cornell.edu/~arb/data/](https://www.cs.cornell.edu/~arb/data/), along with appropriate citations. The function `read_data` in `utils.py` can read these datasets (thanks to Phil Chodrow for the function, which I adapted from his repository [here](https://github.com/PhilChodrow/hypergraph/blob/0e1681f4aa634cb8489cc767f7f144d428be74be/read.py)).

# Help
This README is a work in progress. If you have any questions, open an issue or email me at [larock@maths.ox.ac.uk](mailto:larock@maths.ox.ac.uk).
