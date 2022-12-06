"""
Define functions for update rules.
"""

"""
    Activates all nodes in the hyperedge if more than
    a number k (stored in configurations dict) of other
    nodes are already activated.
"""
def absolute_update(H, edge_id, configuration, t):
    # Only consider inactive edges
    new_activations = 0
    if H.edges[edge_id]["active"] != 1:
        num_active = sum([H.nodes[node]["active"] for node in H.edges.members(edge_id)])
        if num_active >= configuration["active_threshold"]:
            H, new_activations = activate_edge(H, edge_id, t)

    return H, new_activations


"""
    Activates all nodes in the hyperedge if more than
    a number k (stored in configurations dict) of other
    nodes are already activated.
"""
def absolute_update_onehop(H, edge_id, configuration, t):
    # If the edge is not yet active, check if it can be activated
    activate = False
    new_activations = 0
    if H.edges[edge_id]["active"] != 1:
        num_active = sum([H.nodes[node]["active"] for node in H.edges.members(edge_id)])
        if num_active >= configuration["active_threshold"]:
            activate = True

    # If the edge is active
    edges_to_activate = set()
    if activate:
        H, new_activations = activate_edge(H, edge_id, t)
        # ToDo: This creates a second loop over the hyperedge
        # in every activation step, which is kind of annoying
        for node in H.edges.members(edge_id):
            edges_to_activate.update(H.nodes.memberships(node))

    # Activate all of the neighbor edges
    for ne_edge_id in edges_to_activate:
        if H.edges[ne_edge_id]["active"] != 1:
            H, ne_new_activations = activate_edge(H, ne_edge_id, t)
            new_activations += ne_new_activations
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
