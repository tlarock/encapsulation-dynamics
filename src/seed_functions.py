import numpy as np

"""
    Randomly choose num_seeds from all nodes participating
    in 2-node edges.
"""
def twonode_seed(H, configuration):
    num_seeds = configuration["initial_active"]
    activated_nodes_arr = np.random.choice(list(set([node for eid in H.edges
                             for node in H.edges.members(eid)
                             if len(H.edges.members(eid)) == 2])), num_seeds)
    return activated_nodes_arr.tolist()

def biased_seed(H, configuration, inverse=False):
    num_seeds = configuration["initial_active"]
    p = []
    for node in H.nodes:
        avg = np.mean([len(H.edges.members(eid)) for eid in H.nodes.memberships(node)])
        if inverse:
            avg = 1.0 / avg
        p.append(avg)

    p = np.array(p)
    p /= p.sum()
    activated_nodes_arr = np.random.choice(list(H.nodes), p=p, replace=False, size=num_seeds)
    return activated_nodes_arr.tolist()

def inverse_biased_seed(H, configuration):
    return biased_seed(H, configuration, inverse=True)
