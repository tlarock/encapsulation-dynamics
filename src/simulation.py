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
    if verbose:
        print(f"Running {num_sims} simulations of {configuration['steps']} steps on a single cpu.")
    output = dict()
    output["activated_edge_sizes"] = []
    for i in range(configuration["num_simulations"]):
        if verbose:
            print(f"Running simulation {i}.")
        results = run_simulation(hyperedges, configuration, results_only=True)
        for key, vals_arr in results.items():
            if key == "activated_edge_sizes":
                output[key].append(vals_arr)
                continue

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
def run_many_parallel(hyperedges, configuration, ncpus, verbose=False):
    num_sims = configuration["num_simulations"]
    if verbose:
        print(f"Running {num_sims} simulations of {configuration['steps']} steps on {ncpus} cpus.")
    args = []
    for i in range(num_sims):
        args.append((hyperedges, configuration, True))

    with Pool(ncpus, initializer=np.random.seed) as p:
        results_list = p.starmap(run_simulation, args)

    output = dict()
    output["activated_edge_sizes"] = []
    for results in results_list:
        for key, vals_arr in results.items():
            if key == "activated_edge_sizes":
                output[key].append(vals_arr)
                continue

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
        # this will be a list of lists of activated edge size
        # distributions to be compatible with both single edge
        # and simultaneous updates
        "activated_edge_sizes": []
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
            elif configuration["update_name"] in ["subface", "subface-strict", "encapsulation-all", "encapsulation-all-strict"]:
                simultaneous_update_step(H, configuration, results_dict, t, inactive_edge_info, configuration["update_name"])

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
    if configuration["update_name"] in ["subface", "subface-strict"]:
        add_subface_attribute(H, dag_type="super")
    elif configuration["update_name"] in ["encapsulation-all", "encapsulation-all-strict"]:
        add_subface_attribute(H, dag_type="both")
    elif configuration["update_name"] in ["encapsulation-immediate"]:
        add_subface_attribute(H, dag_type="k-1")

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
    the encapsulation DAG in both directions. Option dag_type controls
    which relationships are stored. Since it is actually more efficient
    to store for each smaller hyperedge the hyperedges that it is a subface
    of, meaning option dag_type="super" for superfaces. Including options
    for computing only subfaces or both for potential future use cases.
"""
def add_subface_attribute(H, dag_type="both"):
    assert dag_type in ["super", "sub", "both", "k-1"], f"{dag_type} not a valid option for dag_type."

    sub = False
    sup = False
    sup_limited = False
    if dag_type == "both":
        sub = True
        sup = True
    elif dag_type == "super":
        sup = True
    elif dag_type == "sub":
        sub = True
    elif dag_type == "k-1":
        sup = True
        sup_limited = True

    # Loop over the hyperedges
    for edge_id in H.edges:
        # Initialize the subfaces dict if necessary
        if sub and not ("subfaces" in H.edges[edge_id]):
            H.edges[edge_id]["subfaces"] = set()
        # Initialize the superfaces dict if necessary
        if sup and not ("superfaces" in H.edges[edge_id]):
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
            if sub:
                # If edge_id is already a subface of cand_id, mark as checked
                if "subfaces" in H.edges[cand_id] and edge_id in H.edges[cand_id]["subfaces"]:
                    candidates_checked.add(cand_id)
            if sup:
                # If edge_id is already a superface of cand_id, mark as checked
                if "superfaces" in H.edges[cand_id] and edge_id in H.edges[cand_id]["superfaces"]:
                    candidates_checked.add(cand_id)


            # Skip a candidate if it has already been checked
            if cand_id in candidates_checked:
                continue

            # Check whether the candidate is a sub/superface
            cand = H.edges.members(cand_id)

            # In the sup limited case, I only add DAG superfaces
            # where the difference in size is 1
            if sup_limited:
                if abs(len(edge) - len(cand)) != 1:
                    continue

            # Actually check for encapsulation
            # and include in the appropriate data structure
            if len(edge) > len(cand):
                if is_encapsulated(edge, cand):
                    if sub:
                        H.edges[edge_id]["subfaces"].add(cand_id)

                    if sup:
                        if "superfaces" in H.edges[cand_id]:
                            H.edges[cand_id]["superfaces"].add(edge_id)
                        else:
                            H.edges[cand_id]["superfaces"] = set([edge_id])
            elif len(cand) > len(edge):
                if is_encapsulated(cand, edge):
                    if sup:
                        H.edges[edge_id]["superfaces"].add(cand_id)
                    if sub:
                        if "subfaces" in H.edges[cand_id]:
                            H.edges[cand_id]["subfaces"].add(edge_id)
                        else:
                            H.edges[cand_id]["subfaces"] = set([edge_id])


def count_active_subfaces(H,
                          inactive_edge_info,
                          edge_index_lookup,
                          nodes_as_subfaces=False):
    # Initialize active counts to 0
    inactive_edge_info["active_counts"] = np.zeros(inactive_edge_info["indices"].shape)

    # If all edges are inactive, nothing to compute since all subfaces are
    # inactive
    if inactive_edge_info["edges"].shape[0] == H.num_edges:
        return

    inactive_edges_set = set(inactive_edge_info["edges"])

    # As a special case, we count single nodes as active 0-faces in 2-node edges
    # It is a special case because we don't include individual nodes as
    # hyperedges, otherwise it would work as normal.
    if nodes_as_subfaces:
        for node in H.nodes.filterby_attr("active", 1):
            memberships = H.nodes.memberships(node) - inactive_edge_info["activated_edges"]
            two_node_edges = [sup_id for sup_id in memberships if len(H.edges.members(sup_id)) == 2]
            for sup_id in two_node_edges:
                sup_index = edge_index_lookup[sup_id]
                inactive_edge_info["active_counts"][sup_index] += 1

    # For each active edge, increment the active_count for its superfaces
    for edge_id in H.edges.filterby_attr("active", 1):
        inactive_superfaces = H.edges[edge_id]["superfaces"] - inactive_edge_info["activated_edges"]
        # Constrain single nodes to only activate pairwise hyperedges
        if len(H.edges.members(edge_id)) == 1:
            inactive_superfaces = [sup_id for sup_id in inactive_superfaces if len(H.edges.members(sup_id)) == 2]
        for sup_id in inactive_superfaces:
            sup_index = edge_index_lookup[sup_id]
            inactive_edge_info["active_counts"][sup_index] += 1


def update_active_subface_counts(H,
                                 inactive_edge_info,
                                 edge_indices_to_activate,
                                 newly_active_nodes,
                                 edge_index_lookup,
                                 nodes_as_subfaces=False):
    if nodes_as_subfaces:
        # For every newly active node, update its 2-node edges
        for node in newly_active_nodes:
            for edge_id in H.nodes.memberships(node):
                if edge_id not in inactive_edge_info["activated_edges"] and \
                   len(H.edges.members(edge_id)) == 2:
                    sup_index = edge_index_lookup[edge_id]
                    inactive_edge_info["active_counts"][sup_index] += 1

    # For every newly active edge, update its superfaces
    for edge_index in edge_indices_to_activate:
        edge_id = inactive_edge_info["edges"][edge_index]
        inactive_superfaces = H.edges[edge_id]["superfaces"] - inactive_edge_info["activated_edges"]
        # Constrain single nodes to only activate pairwise hyperedges
        if len(H.edges.members(edge_id)) == 1:
            inactive_superfaces = [sup_id for sup_id in inactive_superfaces if len(H.edges.members(sup_id)) == 2]
        for sup_id in inactive_superfaces:
            sup_index = edge_index_lookup[sup_id]
            inactive_edge_info["active_counts"][sup_index] += 1


def count_active_nodes(H,
                       inactive_edge_info,
                       edge_index_lookup):
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


def update_active_node_counts(H,
                              inactive_edge_info,
                              newly_active_nodes,
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
            count_active_nodes(H, inactive_edge_info, edge_index_lookup)
        elif count_type in ["subface", "encapsulation-all"]:
            count_active_subfaces(H, inactive_edge_info, edge_index_lookup, nodes_as_subfaces=True)
        elif count_type in ["subface-strict", "encapsulation-all-strict"]:
            count_active_subfaces(H, inactive_edge_info, edge_index_lookup, nodes_as_subfaces=False)


        # If down dynamics, compute the threshold
        if configuration["update_name"] == "down":
            inactive_edge_info["thresholds"] = inactive_edge_info["sizes"] - configuration["active_threshold"]
            inactive_edge_info["thresholds"][inactive_edge_info["thresholds"] <= 0] = 1
        elif configuration["update_name"] in ["up", "subface", "subface-strict"]:
            # If up dynamics the threshold is static. This also applies for subface.
            inactive_edge_info["thresholds"] = configuration["active_threshold"]
        elif configuration["update_name"] in ["encapsulation-all", "encapsulation-all-strict"]:
            # If encapsulation-all dynamics the threshold is the number of
            # subfaces that exist in the hypergraph
            inactive_edge_info["thresholds"] = np.array([max(1, len(H.edges[edge_id]["subfaces"]))
                                                         for edge_id in inactive_edge_info["edges"] if edge_id not in inactive_edge_info["activated_edges"]])


    # Get the indices of hyperedges to activate this step
    edge_indices_to_activate = (inactive_edge_info["active_counts"] >= inactive_edge_info["thresholds"]).nonzero()[0]

    # Activate the edges
    results_dict["activated_edge_sizes"].append([])
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
        results_dict["activated_edge_sizes"][-1].append(float(len(H.edges.members(edge_id))))

    # Update the relevant activated counts
    if count_type == "node":
        update_active_node_counts(H,
                                  inactive_edge_info,
                                  newly_active_nodes,
                                  edge_index_lookup)
    elif count_type in ["subface", "encapsulation-all"]:
        update_active_subface_counts(H,
                                     inactive_edge_info,
                                     edge_indices_to_activate,
                                     newly_active_nodes,
                                     edge_index_lookup,
                                     nodes_as_subfaces=True)
    elif count_type in ["subface-strict", "encapsulation-all-strict"]:
        update_active_subface_counts(H,
                                     inactive_edge_info,
                                     edge_indices_to_activate,
                                     newly_active_nodes,
                                     edge_index_lookup,
                                     nodes_as_subfaces=False)

    # Remove the relevant indices from the numpy arrays
    inactive_edge_info["edges"] = np.delete(inactive_edge_info["edges"], edge_indices_to_activate)
    inactive_edge_info["sizes"] = np.delete(inactive_edge_info["sizes"], edge_indices_to_activate)
    inactive_edge_info["active_counts"] = np.delete(inactive_edge_info["active_counts"], edge_indices_to_activate)
    if configuration["update_name"] in ["down", "encapsulation-all", "encapsulation-all-strict"]:
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
    results_dict["activated_edge_sizes"].append([])
    if edge_active_before == 0 and activate:
        H, new_activations = activate_edge(H, edge_id, t)
        results_dict["edges_activated"][t] = 1.0
        results_dict["nodes_activated"][t] = len(new_activations)
        results_dict["activated_edge_sizes"][-1].append(float(len(H.edges.members(edge_id))))

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
