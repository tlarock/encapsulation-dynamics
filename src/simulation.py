import xgi
import numpy as np
from multiprocessing import Pool

def activate_edge(H, edge_id, t):
    H.edges[edge_id]["active"] = 1
    hyperedge = H.edges.members(edge_id)
    new_activations = 0
    for node in hyperedge:
        # Activate the node if it is not already
        if H.nodes[node]["active"] == 0:
            H.nodes[node]["active"] = 1
            H.nodes[node]["activation_time"] = t
            H.nodes[node]["activated_by"] = edge_id
            new_activations += 1
    return H, new_activations

""" To simulate and track the dynamics, I am going to give each node and hyperedge some attributes:
     active: 0 if inactive, 1 if active
     activation_time: If node is active, time in the dynamics when the node became active. Default -1.
     activated_by (node only): ID of the hyperedge that activated the node. Default -1.
"""
def initialize_dynamics(rng, H, configuration):

    # All nodes are given active status 0 to start
    for node in H.nodes:
        H.add_node(node, active = 0, activation_time = -1, activated_by = -1)

    # Seed nodes can either be predetermined, chosen by a function, or chosen at random
    if "seed_activated" in configuration and \
       len(configuration["seed_activated"]) == configuration["initial_active"]:
        activated_nodes = configuration["seed_activated"]
    elif "seed_function" in configuration:
        activated_nodes = configuration["seed_function"](H, configuration)
    else:
        activated_nodes = rng.choice(H.nodes, configuration["initial_active"])

    # Change the active status of the seed nodes
    for node in activated_nodes:
        H.nodes[node]["active"] = 1
        H.nodes[node]["activation_time"] = 0
        # Arbitrary value to indicate which nodes are seeds
        H.nodes[node]["activated_by"] = -10

    # All edges are initially inactive
    edge_attribute_dict = dict()
    for edge_id in H.edges:
        edge_attribute_dict[edge_id] = {
            "active": 0,
            "activation_time":0
        }
    xgi.set_edge_attributes(H, values=edge_attribute_dict)

    return H

def run_simulation(hyperedges, configuration):
    rng = np.random.default_rng()
    H = initialize_dynamics(rng, xgi.Hypergraph(incoming_data=hyperedges), configuration)
    T = configuration["steps"]
    # results_dict contains all of the results for this simulation.
    # Many results can also be computed from H after a simulation, but
    # for convenience I store them separately.
    results_dict = {
        "nodes_activated": np.zeros(T+1),
        "edges_activated": np.zeros(T+1),
        "activated_edge_sizes": np.zeros(T+1)
    }

    # Store the number of initially active nodes
    results_dict["nodes_activated"][0] = len(list(H.nodes.filterby_attr("active", 1)))

    # To speed things up slightly when doing size-biased sampling, I am
    # maintaining three lists in reference to inactive edges:
    # inactive_edges: The actual list of edge IDs
    inactive_edges = np.array(list(H.edges.filterby_attr("active", 0)))
    # _sizes: # of nodes in each edge
    if configuration["selection_name"] in ["uniform", "biased", "simultaneous"]:
        inactive_edges_sizes = np.array([float(len(H.edges.members(edge_id))) for edge_id in
                            inactive_edges])
    elif configuration["selection_name"] == "inverse":
        inactive_edges_sizes = np.array([1.0 / float(len(H.edges.members(edge_id))) for edge_id in
                            inactive_edges])

    sum_of_sizes = inactive_edges_sizes.sum()
    # _indices: List of indices matching across inactive_edges and _sizes
    # This is the list we will actually sample from.
    inactive_edges_indices = np.array(list(range(0, len(inactive_edges))))

    for t in range(1, T+1):
        # Check for saturation
        if len(inactive_edges) == 0:
            break

        if configuration["single_edge_update"]:
            # Choose an edge using selection_function
            edge_index = configuration["selection_function"](rng,
                                                             H,
                                                             inactive_edges_sizes,
                                                             sum_of_sizes,
                                                             inactive_edges_indices
                                                            )
            # Get the edge id in H from the inactive_edges list
            edge_id = inactive_edges[edge_index]

            # Note whether the edge was activte before and how many
            # of its constiuent nodes were already active
            edge_active_before = H.edges[edge_id]["active"]

            # Run the updae function
            activate = configuration["update_function"](H, edge_id, configuration, t)

            # If the edge was activated in this timestep, update the time series
            if edge_active_before == 0 and activate:
                H, new_activations = activate_edge(H, edge_id, t)
                results_dict["edges_activated"][t] = 1.0
                results_dict["nodes_activated"][t] = new_activations
                results_dict["activated_edge_sizes"][t] = float(len(H.edges.members(edge_id)))

                # Remove edge from inactive_edges list by swapping with the final
                # element, then popping the list
                sum_of_sizes -= inactive_edges_sizes[edge_index]
                inactive_edges[edge_index] = inactive_edges[-1]
                inactive_edges_sizes[edge_index] = inactive_edges_sizes[-1]
                inactive_edges = inactive_edges[:-1]
                inactive_edges_sizes = inactive_edges_sizes[:-1]
                inactive_edges_indices = inactive_edges_indices[:-1]
        else:
            edge_indices_to_delete = []
            edge_ids_to_activate = []
            for edge_index in inactive_edges_indices:
                # Get the edge id in H from the inactive_edges list
                edge_id = inactive_edges[edge_index]

                # Note whether the edge was activte before and how many
                # of its constiuent nodes were already active
                edge_active_before = H.edges[edge_id]["active"]

                # Run the updae function
                activate = configuration["update_function"](H, edge_id, configuration, t)

                # If the edge was activated in this timestep, update the time series
                if edge_active_before == 0 and activate:
                    edge_indices_to_delete.append(edge_index)
                    edge_ids_to_activate.append(edge_index)

            # Actually activate the edges
            for edge_id in edge_ids_to_activate:
                H, new_activations = activate_edge(H, edge_id, t)
                results_dict["edges_activated"][t] += 1.0
                results_dict["nodes_activated"][t] += new_activations
                # ToDo FixMe: This is wrong for simultaneous update. Need to
                # make it list-like.
                results_dict["activated_edge_sizes"][t] = float(len(H.edges.members(edge_id)))

            # Remove the relevant indices from the numpy arrays
            inactive_edges = np.delete(inactive_edges, edge_indices_to_delete)
            inactive_edges_sizes = np.delete(inactive_edges_sizes, edge_indices_to_delete)
            # Since we are not swapping and popping as in the single edge case,
            # just need a sequential range here.
            inactive_edges_indices = np.arange(inactive_edges.shape[0])

    return H, results_dict

"""
    Runs multiple simulations on hyperedges using
    settings in configuration. Returns a dictionary
    with matrices of results for node and edge activation.
"""
def run_many_simulations(hyperedges, configuration, verbose=False):
    output = dict()
    for i in range(configuration["num_simulations"]):
        if verbose:
            print(f"Running simulation {i}.")
        H, results = run_simulation(hyperedges, configuration)
        for key, vals_arr in results.items():
            if key not in output:
                output[key] = vals_arr
            else:
                output[key] = np.vstack((output[key], vals_arr))
    return output


"""
    Runs multiple simulations in parallel using multiprocessing
    on hyperedges using settings in configuration. Returns a dictionary
    with matrices of results for node and edge activation.
"""
def run_many_parallel(hyperedges, configuration, ncpus):
    num_sims = configuration["num_simulations"]
    print(f"Running {num_sims} simulations on {ncpus} cpus.")
    args = []
    for i in range(num_sims):
        args.append((hyperedges, configuration))

    with Pool(ncpus, initializer=np.random.seed) as p:
        results_list = p.starmap(run_simulation, args)

    output = dict()
    for _, results in results_list:
        for key, vals_arr in results.items():
            if key not in output:
                output[key] = vals_arr
            else:
                output[key] = np.vstack((output[key], vals_arr))
    return output
