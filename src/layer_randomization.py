import numpy as np
from random import shuffle

"""
    Accepts a list of hyperedges. Returns a randomized list of hyperedges
    with the same size distribution, but with nodes shuffled in each layer.
"""
def layer_randomization(hyperedges):
    rng = np.random.default_rng()
    # Construct mapping from size to hyperedge list
    by_size = dict()
    for he in hyperedges:
        k = len(he)
        if k not in by_size:
            by_size[k] = []
        by_size[k].append(he)

    random_hyperedges = {k:[] for k in by_size.keys()}
    # for each size
    for k in sorted(list(by_size.keys())):
        # get the list of nodes
        nodes = list(set([u for he in by_size[k] for u in he]))
        # shuffle the nodes to create a new mapping
        new_nodes = list(nodes)
        shuffle(new_nodes)
        node_map = {nodes[i]: new_nodes[i] for i in range(len(nodes))}
        # remap the layer
        random_hyperedges[k] = [tuple([node_map[u] for u in he]) for he in by_size[k]]

    return [he for k in random_hyperedges for he in random_hyperedges[k]]
