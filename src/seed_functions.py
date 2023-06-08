import numpy as np


### FUNCTIONS FOR NODE SEEDS ###
"""
    Randomly choose num_seeds nodes from all nodes participating
    in 2-node edges.
"""
def twonode_seed(rng, H, configuration):
    num_seeds = configuration["initial_active"]
    activated_nodes_arr = rng.choice(list(set([node for eid in H.edges
                             for node in H.edges.members(eid)
                             if len(H.edges.members(eid)) == 2])), replace=False, size = num_seeds)
    return activated_nodes_arr.tolist()


"""
    Randomly choose num_seeds nodes with probability proportional to the
    average size of the hyperedges the node participates in.
"""
def biased_seed(rng, H, configuration, inverse=False):
    num_seeds = configuration["initial_active"]
    p = []
    for node in H.nodes:
        avg = np.mean([len(H.edges.members(eid)) for eid in H.nodes.memberships(node)])
        if inverse:
            avg = 1.0 / avg
        p.append(avg)

    p = np.array(p)
    p /= p.sum()
    activated_nodes_arr = rng.choice(list(H.nodes), p=p, replace=False, size=num_seeds)
    return activated_nodes_arr.tolist()

"""
    Wrapper for inverse of the average hyperedge size bias.
"""
def inverse_biased_seed(rng, H, configuration):
    return biased_seed(rng, H, configuration, inverse=True)


"""
    Randomly choose num_seeds nodes with probability proportional to
    the hyperdegree of the node (number of hyperedges the node
    participates in).
"""
def degree_biased_seed(rng, H, configuration, inverse=False):
    num_seeds = configuration["initial_active"]
    p = np.array([float(H.nodes.degree[node]) for node in H.nodes])
    if inverse:
        p = 1.0 / p
    p /= p.sum()
    activated_nodes_arr = rng.choice(list(H.nodes), p=p, replace=False, size=num_seeds)
    return activated_nodes_arr.tolist()

"""
    Wrapper for inverse of the hyperdegree.
"""
def inverse_degree_biased(rng, H, configuration):
    return degree_biased_seed(rng, H, configuration, inverse=True)

### FUNCTIONS FOR EDGE SEEDS ###
"""
    Randomly choose num_seeds edges with probability proportional to the
    size of the edge.
"""
def size_biased_seed(rng, H, configuration, inverse=False):
    num_seeds = configuration["initial_active"]
    p = np.array([float(len(H.edges.members(edge_id))) for edge_id in H.edges])
    if inverse:
        p = 1.0 / p
    p /= p.sum()
    activated_edges_arr = rng.choice(list(H.edges), p=p, replace=False, size=num_seeds)
    return activated_edges_arr.tolist()

"""
    Wrapper for inverse of hyperedge size.
"""
def inverse_size_biased(rng, H, configuration):
    return size_biased_seed(rng, H, configuration, inverse=True)
