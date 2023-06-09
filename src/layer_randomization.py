import numpy as np

"""
    Accepts a list of hyperedges. Returns a randomized list of hyperedges
    with the same size distribution, but with nodes shuffled in each layer.
    """
def layer_randomization(hyperedges, layers=[]):
    rng = np.random.default_rng()

    # Construct mapping from size to hyperedge list
    by_size = dict()
    for he in hyperedges:
        k = len(he)
        if k not in by_size:
            by_size[k] = []
        by_size[k].append(he)

    # Dictionary to store output by size
    random_hyperedges = {k:[] for k in by_size.keys()}

    # Deal with only reshuffling specific layers
    layers = sorted(layers)
    if len(layers) == 0:
        # if no particular layers were given, shuffle all layers
        layers = sorted(list(by_size.keys()))
    else:
        # copy layers that will not be reshuffled into output
        not_reshuffled = set(by_size.keys()) - set(layers)
        for k in not_reshuffled:
            random_hyperedges[k] = by_size[k]

    # for each size that will be reshuffled
    for k in layers:
        # get the list of nodes that apppear in hyperedges of size k
        nodes = list(set([u for he in by_size[k] for u in he]))
        # shuffle the nodes
        new_nodes = list(nodes)
        rng.shuffle(new_nodes)
        # Create a mapping that will relabel the nodes
        node_map = {nodes[i]: new_nodes[i] for i in range(len(nodes))}
        # remap the layer
        random_hyperedges[k] = [tuple([node_map[u] for u in he]) for he in by_size[k]]

    return [he for k in random_hyperedges for he in random_hyperedges[k]]
