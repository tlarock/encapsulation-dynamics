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
           return True

    return False

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
            return True

    return False
