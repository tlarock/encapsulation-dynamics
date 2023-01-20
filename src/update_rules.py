"""
Define functions for update rules.
"""

"""
    up-threshold

    Activates all nodes in the hyperedge if more than
    a number h (stored in configurations dict) of other
    nodes are already activated.
"""
def absolute_update_up(H, edge_id, configuration, t):
    # Only consider inactive edges
    new_activations = 0
    if H.edges[edge_id]["active"] != 1:
        num_active = sum([H.nodes[node]["active"] for node in H.edges.members(edge_id)])
        if num_active >= configuration["active_threshold"]:
            H, new_activations = activate_edge(H, edge_id, t)

    return H, new_activations

"""
    down-threshold

    Activates all nodes in the hyperedge if more than
    a number k-h (stored in configurations dict) of other
    nodes are already activated.
"""
def absolute_update_down(H, edge_id, configuration, t):
    # Only consider inactive edges
    new_activations = 0
    if H.edges[edge_id]["active"] != 1:
        num_active = sum([H.nodes[node]["active"] for node in H.edges.members(edge_id)])
        k = len(H.edges.members(edge_id))
        threshold = k - configuration["active_threshold"]
        # NOTE: I am enforcing a threshold of at least 1 active node
        if threshold <= 0:
            threshold = 1

        if num_active >= threshold:
            H, new_activations = activate_edge(H, edge_id, t)

    return H, new_activations


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
