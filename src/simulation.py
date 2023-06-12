import xgi
import numpy as np
from multiprocessing import Pool
from encapsulation_dag import is_encapsulated

"""
    Runs multiple simulations on hyperedges using
    settings in configuration on a single CPU.
    Returns a dictionary with matrices of results
    for node and edge activation.
"""
def run_many_simulations(hyperedges, configuration, verbose=False):
    np.random.seed()
    num_sims = configuration["num_simulations"]
    print(f"Running {num_sims} simulations of {configuration['steps']} steps on a single cpu.")
    output = dict()
    for i in range(configuration["num_simulations"]):
        if verbose:
            print(f"Running simulation {i}.")
        results = run_simulation(hyperedges, configuration, results_only=True)
        for key, vals_arr in results.items():
            if key not in output:
                output[key] = vals_arr
            else:
                output[key] = np.vstack((output[key], vals_arr))

    output["total_edges"] = len(hyperedges)
    output["total_nodes"] = len(set([u for he in hyperedges for u in he]))

    return output


"""
    Runs multiple simulations in parallel using multiprocessing
    on hyperedges using settings in configuration. Returns a dictionary
    with matrices of results for node and edge activation.
"""
def run_many_parallel(hyperedges, configuration, ncpus):
    num_sims = configuration["num_simulations"]
    print(f"Running {num_sims} simulations of {configuration['steps']} steps on {ncpus} cpus.")
    args = []
    for i in range(num_sims):
        args.append((hyperedges, configuration, True))

    with Pool(ncpus, initializer=np.random.seed) as p:
        results_list = p.starmap(run_simulation, args)

    output = dict()
    for results in results_list:
        for key, vals_arr in results.items():
            if key not in output:
                output[key] = vals_arr
            else:
                output[key] = np.vstack((output[key], vals_arr))

    output["total_edges"] = len(hyperedges)
    output["total_nodes"] = len(set([u for he in hyperedges for u in he]))

    return output


"""
    Main function driving a simulation. Handles initialization and calls the
    _step functions for simulating forward.
"""
def run_simulation(hyperedges, configuration, results_only=False):
    rng = np.random.default_rng()
    # Initialize the xgi.Hypergraph structure using parameters
    # stored in configuration
    H = initialize_dynamics(rng, hyperedges, configuration)

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
    results_dict["edges_activated"][0] = len(list(H.edges.filterby_attr("active", 1)))

    # To speed things up slightly when doing size-biased sampling, I am
    # maintaining three lists in reference to inactive edges:
    inactive_edge_info = dict()
    # inactive_edge_info["edges"]: The actual list of edge IDs
    inactive_edge_info["edges"] = np.array(list(H.edges.filterby_attr("active", 0)))
    # ["sizes"]: # of nodes in each edge
    if configuration["selection_name"] in ["uniform", "biased", "simultaneous"]:
        inactive_edge_info["sizes"] = np.array([float(len(H.edges.members(edge_id))) for edge_id in
                            inactive_edge_info["edges"]])
    elif configuration["selection_name"] == "inverse":
        inactive_edge_info["sizes"] = np.array([1.0 / float(len(H.edges.members(edge_id))) for edge_id in
                            inactive_edge_info["edges"]])

    # ["sum_of_sizes"]: int representing the sum of the sizes array
    inactive_edge_info["sum_of_sizes"] = inactive_edge_info["sizes"].sum()

    # ["indices"]: List of indices matching across inactive_edge_info["edges"]
    # and ["sizes"]. This is the list we will actually sample from.
    inactive_edge_info["indices"] = np.array(list(range(0, len(inactive_edge_info["edges"]))))

    for t in range(1, T+1):
        # Check for saturation
        if inactive_edge_info["edges"].shape[0] == 0:
            break

        if configuration["selection_name"] == "simultaneous":
            if configuration["update_name"] in ["up", "down"]:
                simultaneous_update_step(H, configuration, results_dict, t, inactive_edge_info, "node")
            elif configuration["update_name"] == "subface":
                simultaneous_update_step(H, configuration, results_dict, t, inactive_edge_info, "subface")

            if results_dict["edges_activated"][t] < 1:
                # If no edge was activated this step, the simulation can stop
                break
        else:
            single_edge_step(H, configuration, results_dict, t, rng,
                    inactive_edge_info)

    # This is gross but saves memory in parallel simulations
    if results_only:
        return results_dict
    else:
        return H, results_dict


"""
    Construct the hypergraph and initialize relevant attributes.
     active: 0 if inactive, 1 if active
     activation_time: If node is active, time in the dynamics when the node became active. Default -1.
     activated_by (node only): ID of the hyperedge that activated the node. Default -1.
"""
def initialize_dynamics(rng, hyperedges, configuration):
    # Construct an empty hypergraph
    H = xgi.Hypergraph()

    # Add edges with default attributes
    for edge in hyperedges:
        H.add_edge(edge, active=0, activation_time=-1)

    # For subface simulations, need to
    # compute encapsulation relationships
    if configuration["update_name"] == "subface":
        add_subface_attribute(H)

    # Give nodes default attributes
    for node in H.nodes:
        H.nodes[node]["active"] = 0
        H.nodes[node]["activation_time"] = -1
        H.nodes[node]["activated_by"] = -1

    if configuration["seeding_strategy"] == "node":
        # Seed nodes can either be predetermined, chosen by a function, or chosen at random
        if "seed_activated" in configuration and \
           len(configuration["seed_activated"]) == configuration["initial_active"]:
            activated_nodes = configuration["seed_activated"]
        elif "seed_function" in configuration and configuration["seed_function"] == "uniform":
            activated_nodes = rng.choice(H.nodes, configuration["initial_active"], replace=False)
        else:
            # Use the input function
            activated_nodes = configuration["seed_function"](rng, H, configuration)

        # Change the active status of the seed nodes
        for node in activated_nodes:
            H.nodes[node]["active"] = 1
            H.nodes[node]["activation_time"] = 0
            # Arbitrary value to indicate which nodes are seeds
            H.nodes[node]["activated_by"] = -10

    elif configuration["seeding_strategy"] == "edge":
        if "seed_function" in configuration and configuration["seed_function"] == "uniform":
            activated_edges = rng.choice(H.edges, configuration["initial_active"], replace=False)
        else:
            # Use the input function
            activated_edges = configuration["seed_function"](rng, H, configuration)

        # Activate the seed edges
        for edge_id in activated_edges:
            H, _ = activate_edge(H, edge_id, 0)

    return H

"""
    Add subfaces attribute to the hypergraph. Effectively computes
    the encapsulation DAG in both directions.

    ToDo: Despite the dynamics being driven from subface --> superface,
    I am actually not ever using the subfaces themselves, since it is
    always faster to know the parent. If memory becomes an issue, I can
    stop storing the subfaces (but leaving for now for completeness/in case
    they are convenient later).
"""
def add_subface_attribute(H):
    # Loop over the hyperedges
    for edge_id in H.edges:
        # Initialize the subfaces dict if necessary
        if not ("subfaces" in H.edges[edge_id]):
            H.edges[edge_id]["subfaces"] = set()

        if not ("superfaces" in H.edges[edge_id]):
            H.edges[edge_id]["superfaces"] = set()

        # Get the hyperedge
        edge = H.edges.members(edge_id)

        # Get all potentially encapsulated hyperedges
        candidates = set()
        for node in edge:
            candidates.update(H.nodes.memberships(node))

        # Check all of the candidates once
        candidates_checked = set()
        for cand_id in candidates:
            # If a cand is already encapsulating edge_id, consider it checked
            if "subfaces" in H.edges[cand_id] and edge_id in H.edges[cand_id]["subfaces"]:
                candidates_checked.add(cand_id)

            # Skip a candidate if it has already been checked
            if cand_id in candidates_checked:
                continue

            # Check whether the candidate is a subface
            cand = H.edges.members(cand_id)
            if len(edge) > len(cand):
                if is_encapsulated(edge, cand):
                    H.edges[edge_id]["subfaces"].add(cand_id)
                    if "superfaces" in H.edges[cand_id]:
                        H.edges[cand_id]["superfaces"].add(edge_id)
                    else:
                        H.edges[cand_id]["superfaces"] = set([edge_id])
            elif len(cand) > len(edge):
                if is_encapsulated(cand, edge):
                    H.edges[edge_id]["superfaces"].add(cand_id)
                    if "subfaces" in H.edges[cand_id]:
                        H.edges[cand_id]["subfaces"].add(edge_id)
                    else:
                        H.edges[cand_id]["subfaces"] = set([edge_id])


def count_active_subfaces(inactive_edge_info, H, edge_index_lookup):
    # Initialize active counts to 0
    inactive_edge_info["active_counts"] = np.zeros(inactive_edge_info["indices"].shape)

    # If all edges are inactive, nothing to compute since all subfaces are
    # inactive
    if inactive_edge_info["edges"].shape[0] == H.num_edges:
        return

    inactive_edges_set = set(inactive_edge_info["edges"])
    for edge_id in H.edges.filterby_attr("active", 1):
        inactive_superfaces = H.edges[edge_id]["superfaces"] - inactive_edge_info["activated_edges"]
        for sup_id in inactive_superfaces:
            sup_index = edge_index_lookup[sup_id]
            inactive_edge_info["active_counts"][sup_index] += 1


def update_active_subface_counts(H, inactive_edge_info,
                                 edge_indices_to_activate, edge_index_lookup):
    for edge_index in edge_indices_to_activate:
        edge_id = inactive_edge_info["edges"][edge_index]
        inactive_superfaces = H.edges[edge_id]["superfaces"] - inactive_edge_info["activated_edges"]
        for sup_id in inactive_superfaces:
            sup_index = edge_index_lookup[sup_id]
            inactive_edge_info["active_counts"][sup_index] += 1


def count_active_nodes(inactive_edge_info, H, edge_index_lookup):
    # Initialize active counts to 0
    inactive_edge_info["active_counts"] = np.zeros(inactive_edge_info["indices"].shape)

    inactive_edges_set = set(inactive_edge_info["edges"])

    # Loop over every active node
    for node in H.nodes.filterby_attr("active", 1):
        # Get the set of inactive edges the node participates in
        memberships = set(H.nodes.memberships(node))
        inactive_memberships = memberships.intersection(inactive_edges_set)
        # Increment the number of active nodes for the edge id
        for edge_id in inactive_memberships:
            edge_index = edge_index_lookup[edge_id]
            inactive_edge_info["active_counts"][edge_index] += 1


def update_active_node_counts(H, inactive_edge_info, newly_active_nodes,
                              edge_index_lookup):
    for node in newly_active_nodes:
        for edge_id in H.nodes.memberships(node):
            # Only update edges that are not about to be
            # deleted to avoid wasted computation
            if edge_id not in inactive_edge_info["activated_edges"]:
                edge_index = edge_index_lookup[edge_id]
                inactive_edge_info["active_counts"][edge_index] += 1


"""
    Model step definition for simultaneous update over all inactive edges.
"""
def simultaneous_update_step(H, configuration, results_dict, t,
                             inactive_edge_info, count_type):
    # Since we are not swapping and popping as in the single edge case,
    # just need a sequential range here.
    inactive_edge_info["indices"] = np.arange(inactive_edge_info["edges"].shape[0])
    # Resetting edge_index_lookup
    edge_index_lookup = dict(zip(inactive_edge_info["edges"], inactive_edge_info["indices"]))

    if t == 1:
        # Store the set of activated edges
        inactive_edge_info["activated_edges"] = set(H.edges.filterby_attr("active", 1))

        # Count the number of active substructures in each inactive edge
        if count_type == "node":
            count_active_nodes(inactive_edge_info, H, edge_index_lookup)
        elif count_type == "subface":
            count_active_subfaces(inactive_edge_info, H, edge_index_lookup)

        # If down dynamics, compute the threshold
        if configuration["update_name"] == "down":
            inactive_edge_info["thresholds"] = inactive_edge_info["sizes"] - configuration["active_threshold"]
            inactive_edge_info["thresholds"][inactive_edge_info["thresholds"] <= 0] = 1
        elif configuration["update_name"] in ["up", "subface"]:
            # If up dynamics the threshold is static. This also applies for subface.
            inactive_edge_info["thresholds"] = configuration["active_threshold"]


    # Get the indices of hyperedges to activate this step
    edge_indices_to_activate = (inactive_edge_info["active_counts"] >= inactive_edge_info["thresholds"]).nonzero()[0]

    # Activate the edges
    newly_active_nodes = set()
    for edge_index in edge_indices_to_activate:
        edge_id = inactive_edge_info["edges"][edge_index]
        inactive_edge_info["activated_edges"].add(edge_id)
        H, new_activations = activate_edge(H, edge_id, t)
        newly_active_nodes.update(new_activations)
        results_dict["edges_activated"][t] += 1.0
        results_dict["nodes_activated"][t] += len(new_activations)
        # ToDo FixMe: This is wrong for simultaneous update. Need to
        # make it list-like.
        results_dict["activated_edge_sizes"][t] = float(len(H.edges.members(edge_id)))

    # Update the relevant activated counts
    if count_type == "node":
        update_active_node_counts(H, inactive_edge_info, newly_active_nodes, edge_index_lookup)
    elif count_type == "subface":
        update_active_subface_counts(H, inactive_edge_info, edge_indices_to_activate, edge_index_lookup)

    # Remove the relevant indices from the numpy arrays
    inactive_edge_info["edges"] = np.delete(inactive_edge_info["edges"], edge_indices_to_activate)
    inactive_edge_info["sizes"] = np.delete(inactive_edge_info["sizes"], edge_indices_to_activate)
    inactive_edge_info["active_counts"] = np.delete(inactive_edge_info["active_counts"], edge_indices_to_activate)
    if configuration["update_name"] == "down":
        inactive_edge_info["thresholds"] = np.delete(inactive_edge_info["thresholds"], edge_indices_to_activate)


"""
    Model step definition for single edge per timestep update.
"""
def single_edge_step(H, configuration, results_dict, t, rng, inactive_edge_info):
    # Choose an edge using selection_function
    edge_index = configuration["selection_function"](rng,
                                                     H,
                                                     inactive_edge_info["sizes"],
                                                     inactive_edge_info["sum_of_sizes"],
                                                     inactive_edge_info["indices"]
                                                    )
    # Get the edge id in H from the inactive_edge_info["edges"] list
    edge_id = inactive_edge_info["edges"][edge_index]

    # Note whether the edge was activte before and how many
    # of its constiuent nodes were already active
    edge_active_before = H.edges[edge_id]["active"]

    # Run the updae function
    activate = configuration["update_function"](H, edge_id, configuration, t)

    # If the edge was activated in this timestep, update the time series
    if edge_active_before == 0 and activate:
        H, new_activations = activate_edge(H, edge_id, t)
        results_dict["edges_activated"][t] = 1.0
        results_dict["nodes_activated"][t] = len(new_activations)
        results_dict["activated_edge_sizes"][t] = float(len(H.edges.members(edge_id)))

        # Remove edge from inactive_edge_info["edges"] list by swapping with the final
        # element, then popping the list
        inactive_edge_info["sum_of_sizes"] -= inactive_edge_info["sizes"][edge_index]
        inactive_edge_info["edges"][edge_index] = inactive_edge_info["edges"][-1]
        inactive_edge_info["sizes"][edge_index] = inactive_edge_info["sizes"][-1]
        inactive_edge_info["edges"] = inactive_edge_info["edges"][:-1]
        inactive_edge_info["sizes"] = inactive_edge_info["sizes"][:-1]
        inactive_edge_info["indices"] = inactive_edge_info["indices"][:-1]


"""
    Activate an edge in the hypergraph and return
    the number of newly activated nodes.
"""
def activate_edge(H, edge_id, t):
    H.edges[edge_id]["active"] = 1
    hyperedge = H.edges.members(edge_id)
    new_activations = set()
    for node in hyperedge:
        # Activate the node if it is not already
        if H.nodes[node]["active"] == 0:
            H.nodes[node]["active"] = 1
            H.nodes[node]["activation_time"] = t
            H.nodes[node]["activated_by"] = edge_id
            new_activations.add(node)
    return H, new_activations
