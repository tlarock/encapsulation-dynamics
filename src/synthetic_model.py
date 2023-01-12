import numpy as np
from itertools import chain, combinations
from collections import Counter

"""
    Accepts a list-like of nodes and constructs
    a minimally connected random graph on the nodes.
"""
def minimally_connected_pairs(nodes):
    needs_connecting = list(nodes)
    component = [np.random.choice(nodes).tolist()]
    needs_connecting.remove(component[-1])
    edges = []
    while len(component) != len(nodes):
        # Choose a random node from component
        cnode = np.random.choice(component)
        # Choose a random node from needs_connecting
        ncnode = np.random.choice(needs_connecting)
        # Add the connection
        edges.append(tuple(sorted([cnode, ncnode])))
        # Remove ncnode from needs_connecting and add to component
        component.append(ncnode)
        needs_connecting.remove(ncnode)
    return edges

"""
    Accept a list of disjoint hyperedges made up of _groups_ and add
    connecting_edges edges between the groups.

    If connecting_edges is less than len(groups)-1, then the groups will
    not be connected. Instead the edges will be placed between the maximum
    number of groups.

    If connecting_edges == len(groups)-1, the edges will be placed such that
    the groups are minimally connected (e.g. for each pair of groups there will be
    a path between them).

    If connecting_edges > len(groups)-1, edges will first be placed such that
    the groups are minimally connected, then the remaining edges will be placed
    randomly (including between already-connected groups).
"""
def add_connecting_hyperedges(hyperedges, groups, connecting_edges):
    # Add connecting hyperedges
    if len(groups) > 1 and connecting_edges > 0:
        group_pairs = list(combinations(list(groups.keys()), 2))
        pairs_indices = list(range(len(group_pairs)))
        if connecting_edges >= len(groups)-1:
            # get a minimally connected graph of the pairs
            minimally_connected = minimally_connected_pairs(list(groups.keys()))
            pair_connection_ids = [group_pairs.index(e) for e in minimally_connected]
            if connecting_edges > len(groups)-1:
                # Add extra connection randomly
                pair_connection_ids += np.random.choice(pairs_indices, connecting_edges-len(pair_connection_ids), replace=True).tolist()
        else:
            # Choose #groups-1 groups to connect randomly
            groups_to_connect = np.random.choice(list(groups.keys()), connecting_edges-1).tolist()
            # Minimally connect the groups
            minimally_connected = minimally_connected_pairs(groups_to_connect)
            pair_connection_ids = [group_pairs.index(e) for e in minimally_connected]

        for idx in pair_connection_ids:
            i,j = group_pairs[idx]
            u = np.random.choice(groups[i], 1)[0]
            v = np.random.choice(groups[j], 1)[0]
            hyperedge = list(sorted([u,v]))
            while hyperedge in hyperedges:
                u = np.random.choice(groups[i], 1)[0]
                v = np.random.choice(groups[j], 1)[0]
                hyperedge = list(sorted([u,v]))
            hyperedges.append(tuple(hyperedge))

    return hyperedges

def add_random_connections(hyperedges, num_random, nodes, hyperedges_nodeset):
    if num_random > 0:
        remaining_nodes = [u for u in nodes if u not in hyperedges_nodeset]
        random_nodes = np.random.choice(remaining_nodes, num_random, replace=False).tolist()
        for node in random_nodes:
            # Randomly choose a partner
            # ToDo: Could randomly choose a hyperedge; this would potentially add _more_ encapsulation
            partner = np.random.choice(list(hyperedges_nodeset))
            hyperedges.append(tuple([node, partner]))
    return hyperedges

"""
    Constructs a hypergraph that is a combination of simplicial complexes, as
    well as a random hypergraph with the same number of nodes, edge size distribution,
    and connectivity.

    The parameter group_sizes is a list-like of integers > 3. Each group
    will be a disjoint set of nodes from 1,...,N, thus N should be >= sum(group_sizes).
    All subsets of size > 1 node for each group will also be added as hyperedges.

    If sum(group_sizes) > N, the remaining nodes will be attached randomly to nodes within
    the groups in size-2 hyperedges.
    # ToDo: This can be generalized to varying noise edge size.

    The paramter connecting_edges is an integer that must be >= len(group_sizes) for the
    resulting hypergraph to be connected (this is not enforced). Connections are added by choosing
    two groups, then adding a size-2 hyperedge by choosing a random node from each group.
    # ToDo: This can be generalized to (a) varying connecting edge size and (b) connecting more
    than two groups at a time.


"""
def synthetic_encapsulation_model(N, group_sizes, connecting_edges):
    assert min(group_sizes) > 2, f"Minimum group size is 3. Found {min(group_sizes)}."
    assert N >= sum(group_sizes), f"N must be >= sum(group_sizes), otherwise groups cannot be formed. Found {N}, {sum(group_sizes)}."

    nodes = list(range(N))

    # Get random disjoint sets of nodes following group_sizes
    sample_nodes = list(nodes)
    groups = dict()
    hyperedges = []
    hyperedges_nodeset = set()
    for group_id, group_size in enumerate(group_sizes):
        # Construct a simplicial complex
        complex_nodes = np.random.choice(sample_nodes, group_size, replace=False).tolist()
        simplicial_complex = [tuple(complex_nodes)]
        simplicial_complex += [tuple(sorted(edge)) for edge in chain.from_iterable(combinations(complex_nodes, r)
                                                                          for r in range(2, len(complex_nodes)))]

        groups[group_id] = complex_nodes
        hyperedges += list(simplicial_complex)
        for u in complex_nodes:
            sample_nodes.remove(u)
            hyperedges_nodeset.add(u)

    # Randomly connect hyperedges
    hyperedges = add_connecting_hyperedges(hyperedges, groups, connecting_edges)

    # Randomly add extra hyperedges
    num_random = N - sum(group_sizes)
    hyperedges = add_random_connections(hyperedges, num_random, nodes, hyperedges_nodeset)

    nodes_in_hyperedges = len(set([u for e in hyperedges for u in e]))
    assert nodes_in_hyperedges == N

    return hyperedges

"""
    Constructs a hypergraph that is effectively a noisy single-component simplicial complex, as
    well as a random hypergraph with the same number of nodes, edge size distribution,
    and connectivity. The procedure for the noisy simplicial complex is as follows:

        1. Choose a set complex_nodes with size k from N.
        2. Create a hyperedge with complex_nodes as members.
        3. Add hyperedges corresponding to all subsets of complex_nodes with
            at least 2 nodes.
        4. For each of the remaining N-k nodes, pair the node
            with a node from complex_nodes to form "noise" hyperedges.
"""
def construct_noisy_complex(N, k):
    nodes = list(range(N))
    # Construct a simplicial complex
    complex_nodes = np.random.choice(nodes, k, replace=False).tolist()
    simplicial_complex = [tuple(complex_nodes)]
    simplicial_complex += [tuple(sorted(edge)) for edge in chain.from_iterable(combinations(complex_nodes, r)
                                                                      for r in range(2, len(complex_nodes)))]

    hyperedges = list(simplicial_complex)
    # Randomly add more hyperedges
    # For now will just add edges
    num_random = N - k
    remaining_nodes = [u for u in nodes if u not in complex_nodes]
    random_nodes = np.random.choice(remaining_nodes, num_random, replace=False).tolist()
    for node in random_nodes:
        # Randomly choose a partner
        # ToDo: Could randomly choose a hyperedge; this would potentially add _more_ encapsulation
        partner = np.random.choice(complex_nodes)
        hyperedges.append(tuple([node, partner]))

    nodes_in_hyperedges = len(set([u for e in hyperedges for u in e]))
    assert nodes_in_hyperedges == k + num_random

    return hyperedges

"""
    Constructs a random hypergraph based on the hypergraph
    stored in hyperedges. Has the same number of edges and
    edge size distribution, but nodes are assigned to hyperedges
    uniformly at random.

    NOTE: This function does _not_ gaurantee that
    the resulting hypergraph is connected!!
"""
def construct_random_hyperedges(hyperedges):
    # Random hyperedges of the same sizes
    hyperedge_nodes = list(set([u for e in hyperedges for u in e]))
    sizes = Counter([len(e) for e in hyperedges])
    nodes_in_random = 0
    while nodes_in_random != len(hyperedge_nodes):
        random_hyperedges = []
        for size in sizes:
            edges = set()
            for _ in range(sizes[size]):
                edge = tuple(np.random.choice(hyperedge_nodes, size, replace=False).tolist())
                while edge in edges:
                    edge = tuple(np.random.choice(hyperedge_nodes, size, replace=False).tolist())
                edges.add(edge)
            for e in edges:
                random_hyperedges.append(e)
        nodes_in_random = len(set([u for e in random_hyperedges for u in e]))
    return random_hyperedges
