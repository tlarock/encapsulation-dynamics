import xgi
import numpy as np

""" To simulate and track the dynamics, I am going to give each node and hyperedge some attributes:
     active: 0 if inactive, 1 if active
     activation_time: If node is active, time in the dynamics when the node became active. Default -1.
     activated_by (node only): ID of the hyperedge that activated the node. Default -1.
"""
def initialize_dynamics(H, configuration):

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
        activated_nodes = np.random.choice(H.nodes, configuration["initial_active"])

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


def run_simulation(H, configuration):
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
    inactive_edges = list(H.edges.filterby_attr("active", 0))
    # _sizes: # of nodes in each edge
    inactive_edges_sizes = [float(len(H.edges.members(edge_id))) for edge_id in inactive_edges]
    # _indices: List of indices matching across inactive_edges and _sizes
    # This is the list we will actually sample from.
    inactive_edges_indices = list(range(0, len(inactive_edges)))

    for t in range(1, T+1):
        # Check for saturation
        if len(inactive_edges) == 0:
            break

        # Choose an edge using selection_function
        edge_index = configuration["selection_function"](H,
                                                      inactive_edges_sizes,
                                                      inactive_edges_indices
                                                     )
        # Get the edge id in H from the inactive_edges list
        edge_id = inactive_edges[edge_index]

        # Note whether the edge was activte before and how many
        # of its constiuent nodes were already active
        edge_active_before = H.edges[edge_id]["active"]

        # Run the updae function
        H, new_activations = configuration["update_function"](H, edge_id, configuration, t)

        # If the edge was activated in this timestep, update the time series
        if edge_active_before == 0 and H.edges[edge_id]["active"] == 1:
            results_dict["edges_activated"][t] = 1.0
            results_dict["nodes_activated"][t] = new_activations
            results_dict["activated_edge_sizes"][t] = float(len(H.edges.members(edge_id)))

            # Remove edge from inactive_edges list by swapping with the final
            # element, then popping the list
            inactive_edges[edge_index] = inactive_edges[-1]
            inactive_edges_sizes[edge_index] = inactive_edges_sizes[-1]
            inactive_edges.pop()
            inactive_edges_sizes.pop()
            inactive_edges_indices.pop()

    return H, results_dict

"""
    Runs multiple simulations on hyperedges using
    settings in configuration. Returns a dictionary
    with matrices of results for node and edge activation.
"""
def run_many_simulations(hyperedges, configuration):
    output = dict()
    for i in range(configuration["num_simulations"]):
        H = xgi.Hypergraph(incoming_data=hyperedges)
        H = initialize_dynamics(H, configuration)
        H, results = run_simulation(H, configuration)
        for key, vals_arr in results.items():
            if key not in output:
                output[key] = vals_arr
            else:
                output[key] = np.vstack((output[key], vals_arr))
    return output
