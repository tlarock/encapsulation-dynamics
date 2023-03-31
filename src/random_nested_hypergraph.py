import numpy as np
from itertools import combinations
from collections import defaultdict

"""
    Sample a hyperedge of size _size_ from _nodes_ that is not
    already present in the set _hyperedges_
"""
def uniform_sample(nodes, size, hyperedges, rng):
    he = tuple(sorted(rng.choice(nodes, size, replace=False)))
    while he in hyperedges:
        he = tuple(sorted(rng.choice(nodes, size, replace=False)))

    return he

"""
    Rewires a hyperedge based on a given pivot node. Accepts the full set of nodes,
    a pivot node that should be included in the new hyperedge, the hyperedge
    being rewired, the set of all hyperedges, and a dictionary mapping a
    hyperedge to all of the nodes above it in any facet.
"""
def rewire_from_pivot(nodeset, pivot, he, hyperedges, facet_nodes, rng):
    new_he = tuple(sorted([pivot] + rng.choice(list(nodeset-facet_nodes[he]), len(he)-1, replace=False).tolist()))
    while new_he in hyperedges:
        new_he = tuple(sorted([pivot] + rng.choice(list(nodeset-facet_nodes[he]), len(he)-1, replace=False).tolist()))
    return new_he

"""
    Random nested facet model. Take a number of nodes N, maximum-size
    hyperedge max_size, number of max_size hyperedges H, and hyperedge-nestedness
    parameter epsilon. Return a hypergraph where there are H hyperedges of
    size s_m and sub hyperedges (size s < s_m) have been rewired with probability
    1-epsilon.
"""
def random_nested_hypergraph(N, max_size, H, epsilons, max_size_overlap=-1):
    rng = np.random.default_rng()
    nodes = list(range(N))
    hyperedges = set()
    facet_nodes = defaultdict(set)
    # For rewiring, we need to keep track of all nodes that
    # are part of a superset of every hyperedge
    if max_size_overlap < 0:
        while len(hyperedges) < H:
            # Choose a uniform random set that is not already present
            he = uniform_sample(nodes, max_size, hyperedges, rng)
            hyperedges.add(he)
    else:
        hyperedges = set(get_overlapping_hyperedges(N, max_size, H, max_size_overlap))

    # add all facets
    for he in set(hyperedges):
        for size in range(2, max_size):
            for subset in combinations(he, size):
                hyperedges.add(subset)
                facet_nodes[subset].update(he)


    # Randomly rewire nested hyperedges with probability 1-epsilon
    nodeset = set(nodes)
    to_remove = set()
    to_add = set()
    for he in hyperedges:
        if len(he) == max_size:
            continue

        if rng.random() < 1-epsilons[len(he)]:
            # Mark this hyperedge for removal
            to_remove.add(he)
            # Choose a pivot node
            pivot = rng.choice(list(he))
            # Rewire the hyperedge
            new_he = rewire_from_pivot(nodeset, pivot, he, hyperedges,
                                       facet_nodes, rng)
            to_add.add(new_he)

    hyperedges -= to_remove
    hyperedges.update(to_add)
    return list(hyperedges)


"""
    Accepts a list of hyperedges created by random_nested_hypergraph
    and a dictionary of deletion_probibility for each size s < max.
    Returns a list of hyperedges where any hyperedge of size s < s_m
    has been removed with probability deletion_prob[s].
"""
def layer_hyperedge_deletion(hyperedges, deletion_probs, rng):
    max_size = max([len(he) for he in hyperedges])
    indices_to_remove = set()
    for idx, he in enumerate(hyperedges):
        if len(he) < max_size:
            q = rng.random()
            if q < deletion_probs[len(he)]:
                indices_to_remove.add(idx)

    return [he for idx, he in enumerate(hyperedges) if idx not in indices_to_remove]


"""
    Accepts a list of hyperedges created by random_nested_hypergraph
    and a deletion_probibility. Returns a list of hyperedges where
    any hyperedge of size smaller than the maximum has been removed
    with probability deletion_prob.
"""
def global_hyperedge_deletion(hyperedges, deletion_prob, rng):
    max_size = max([len(he) for he in hyperedges])
    indices_to_remove = set()
    for idx, he in enumerate(hyperedges):
        if len(he) < max_size:
            q = rng.random()
            if q < deletion_prob:
                indices_to_remove.add(idx)

    return [he for idx, he in enumerate(hyperedges) if idx not in indices_to_remove]


def get_overlapping_hyperedges(N, maximum_size, num_hyperedges, overlap):
    if overlap == maximum_size:
        print(f"Overlap and maximum_size should be different, found {overlap}, {maximum_size}.")
        return []
    if overlap == 0:
        hyperedges = [tuple(list(range(i, i+maximum_size))) for i in range(0, N, maximum_size)]
        while len(hyperedges) > num_hyperedges:
            hyperedges.pop()
    else:
        # Get the nodes
        nodes = list(range(N))
        nodeset = set(nodes)
        # choose a first hyperedge
        hyperedges = [tuple(sorted(np.random.choice(nodes, maximum_size, replace=False).tolist()))]
        for _ in range(num_hyperedges-1):
            # choose two nodes from a previous hyperedge
            overlap_nodes = np.random.choice(hyperedges[np.random.randint(0, high=len(hyperedges))], overlap, replace=False).tolist()
            # Choose two random nodes
            random_nodes = np.random.choice(list(nodeset-set(overlap_nodes)), maximum_size-overlap, replace=False).tolist()
            hyperedge = tuple(sorted(overlap_nodes + random_nodes))
            while hyperedge in hyperedges or any([len(set(hyperedge).intersection(set(he))) > overlap for he in hyperedges]):
                # choose two nodes from a previous hyperedge
                overlap_nodes = np.random.choice(hyperedges[np.random.randint(0, high=len(hyperedges))], overlap, replace=False).tolist()
                # Choose two random nodes
                random_nodes = np.random.choice(list(nodeset-set(overlap_nodes)), maximum_size-overlap, replace=False).tolist()
                hyperedge = tuple(sorted(overlap_nodes + random_nodes))
            hyperedges.append(hyperedge)
    return hyperedges
