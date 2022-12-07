import xgi
import numpy as np
import sys
sys.path.append("../src/")
from utils import read_data, read_random_hyperedges
from update_rules import *
from selection_rules import *
from simulation import *
from plot_simulation_results import *


# Use my own code to get the list of hyperedges from Austin's format
dataset_name = "contact-high-school"
#dataset_name = "contact-primary-school"
dataset = f"../data/{dataset_name}/{dataset_name}-"

#dataset_name = "coauth-MAG-Geology-full"
#dataset = f"../../HAD.jl/had_data/{dataset_name}/{dataset_name}-"

hyperedges = read_data(dataset, multiedges=False)

random_path = f"../data/{dataset_name}/"
#random_path = f"../../HAD.jl/had_data/{dataset_name}/"

random_hyperedges = read_random_hyperedges(random_path + "randomizations/random-simple-0.txt")

results_path = f"../results/{dataset_name}/{dataset_name}"


#configuration = {
#    "initial_active": 10_000,
#    "steps": 5000,
#    "active_threshold": 1,
#    "num_simulations": 5
#}

configuration = {
    "initial_active": 1,
    "steps": 5000,
    "active_threshold": 1,
    "num_simulations": 100
}

# Dictionary of configuration/parameters
print("Running uniform up")
configuration["selection_function"] = uniform_inactive
configuration["update_function"] = absolute_update_up

output_obs = run_many_simulations(hyperedges, configuration)
output_rnd = run_many_simulations(random_hyperedges, configuration)
plot_cumulative_averages(configuration, output_obs, output_rnd)
plot_filename = results_path
plot_filename += f"_uniform_up-{configuration['active_threshold']}"
plot_filename += f"_runs-{configuration['num_simulations']}"
plt.tight_layout()
plt.savefig(plot_filename + ".pdf", dpi=150)


# Dictionary of configuration/parameters
print("Running uniform down")
configuration["selection_function"] = uniform_inactive
configuration["update_function"] = absolute_update_down
output_obs = run_many_simulations(hyperedges, configuration)
output_rnd = run_many_simulations(random_hyperedges, configuration)
plot_cumulative_averages(configuration, output_obs, output_rnd)
plot_filename = results_path
plot_filename += f"_uniform_down-{configuration['active_threshold']}"
plot_filename += f"_runs-{configuration['num_simulations']}"
plt.tight_layout()
plt.savefig(plot_filename + ".pdf", dpi=150)


# Dictionary of configuration/parameters
print("Running biased up")
configuration["selection_function"] = biased_inactive
configuration["update_function"] = absolute_update_up
output_obs = run_many_simulations(hyperedges, configuration)
output_rnd = run_many_simulations(random_hyperedges, configuration)
plot_cumulative_averages(configuration, output_obs, output_rnd)
plot_filename = results_path
plot_filename += f"_biased_up-{configuration['active_threshold']}"
plot_filename += f"_runs-{configuration['num_simulations']}"
plt.tight_layout()
plt.savefig(plot_filename + ".pdf", dpi=150)


# Dictionary of configuration/parameters
print("Running biased down")
configuration["selection_function"] = biased_inactive
configuration["update_function"] = absolute_update_down
output_obs = run_many_simulations(hyperedges, configuration)
output_rnd = run_many_simulations(random_hyperedges, configuration)
plot_cumulative_averages(configuration, output_obs, output_rnd)
plot_filename = results_path
plot_filename += f"_biased_down-{configuration['active_threshold']}"
plot_filename += f"_runs-{configuration['num_simulations']}"
plt.tight_layout()
plt.savefig(plot_filename + ".pdf", dpi=150)
