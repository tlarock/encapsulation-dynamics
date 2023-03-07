import numpy as np
from itertools import combinations
from collections import defaultdict


"""
    Sample a hyperedge of size _size_ from _nodes_ that is not
    already present in the set _hyperedges_
"""
def uniform_sample(nodes, size, hyperedges):
    he = tuple(sorted(np.random.choice(nodes, size, replace=False)))
    while he in hyperedges:
        he = tuple(sorted(np.random.choice(nodes, size, replace=False)))

    return he

"""
"""
def rewire_from_pivot(nodeset, pivot, he, hyperedges, facet_nodes):
    new_he = tuple(sorted([pivot] + np.random.choice(list(nodeset-facet_nodes[he]), len(he)-1, replace=False).tolist()))
    while new_he in hyperedges:
        new_he = tuple(sorted([pivot] + np.random.choice(list(nodeset-facet_nodes[he]), len(he)-1, replace=False).tolist()))
    return new_he

"""
    Random nested facet model. Take a number of nodes N, maximum-size
    hyperedge max_size, number of max_size hyperedges H, and hyperedge-nestedness
    parameter epsilon. Return a hypergraph where there are H hyperedges of
    size s_m and sub hyperedges (size s < s_m) have been rewired with probability
    1-epsilon.
"""
def random_nested_hypergraph(N, max_size, H, epsilons):
    nodes = list(range(N))
    hyperedges = set()
    max_size_edges = 0
    facet_nodes = defaultdict(set)
    while max_size_edges < H:
        # Choose a uniform random set that is not already present
        he = uniform_sample(nodes, max_size, hyperedges)
        hyperedges.add(he)
        max_size_edges += 1
        # add all facets
        for size in range(2, max_size):
            for subset in combinations(he, size):
                hyperedges.add(subset)
                facet_nodes[subset].update(he)

    # Randomly rewire nested hyperedges with probability 1-epsilon
    nodeset = set(nodes)
    to_remove = set()
    to_add = set()
    for he in hyperedges:
        if len(he) < max_size and np.random.random() < 1-epsilons[len(he)]:
            # Mark this hyperedge for removal
            to_remove.add(he)
            # Choose a pivot node
            pivot = np.random.choice(list(he))
            # Rewire the hyperedge
            new_he = rewire_from_pivot(nodeset, pivot, he, hyperedges, facet_nodes)
            to_add.add(new_he)

    hyperedges -= to_remove
    hyperedges.update(to_add)
    return list(hyperedges)
