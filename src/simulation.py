import numpy as np

""" To simulate and track the dynamics, I am going to give each node and hyperedge some attributes:
     active: 0 if inactive, 1 if active
     activation_time: If node is active, time in the dynamics when the node became active. Default -1.
     activated_by (node only): ID of the hyperedge that activated the node. Default -1.
"""
def initialize_dynamics(H, configuration):
    for node in H.nodes:
        H.add_node(node, active = 0, activation_time = -1, activated_by = -1)

    activated_nodes = np.random.choice(H.nodes, configuration["initial_active"])
    for node in activated_nodes:
        H.nodes[node]["active"] = 1
        H.nodes[node]["activation_time"] = 0
        H.nodes[node]["activated_by"] = -10

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
    results_dict = {
        "nodes_activated": np.zeros(T),
        "edges_activated": np.zeros(T)
    }
    for t in range(T):
        # Check for saturation
        if len(H.edges.filterby_attr("active", 0)) == 0:
            break

        # Choose an edge using selection_function
        edge_id = configuration["selection_function"](H)

        # Note whether the edge was activte before and how many
        # of its constiuent nodes were already active
        edge_active_before = H.edges[edge_id]["active"]

        # Run the updae function
        H, new_activations = configuration["update_function"](H, edge_id, configuration, t)

        # If the edge was activated in this timestep, update the time series
        if edge_active_before == 0 and H.edges[edge_id]["active"] == 1:
            results_dict["edges_activated"][t] = 1.0
            results_dict["nodes_activated"][t] = new_activations
    return H, results_dict
