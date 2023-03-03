import numpy as np
from collections import defaultdict

def sample_hyperedges(nodes, size, size_distribution, node_probabilities):
    uniform_sample = set()
    degrees = {node: 0 for node in nodes}
    for _ in range(size_distribution[size]):
        he = tuple(sorted(np.random.choice(nodes, size, replace=False, p=node_probabilities)))
        while he in uniform_sample:
            he = tuple(sorted(np.random.choice(nodes, size, replace=False, p=node_probabilities)))
        uniform_sample.add(he)
        for node in he:
            degrees[node] += 1

    return uniform_sample, degrees


"""
    Construct a random hypergraph on N nodes with a given hyperedge size distribution
    with one of three options for correlation:
        1. uncorrelated: Choose hyperedges of all sizes uniformly at random
        2. positive: Choose hyperedges based on degree of nodes in larger hyperedges.
        3. mixed: Choose hyperedges based on inverse degree of nodes in larger hyperedges. Mixed
            because every layer flips the correlation.
        4. negative: Choose hyperedges based on inverse degree of nodes in largest hyperedges.

    Positively correlated hypergraphs should have more encapsulation relationships; negatively
    correlated should have fewer; and uniform should fall in-between, but also with very few.

    size_distribution: dictionary with keys of sizes and values num hyperedges of that size
"""
def random_degree_hypergraph(N, size_distribution, correlation="uncorrelated",
                            first_N=0):
    assert correlation in ["uncorrelated", "positive", "negative", "mixed"], \
            f"{correlation} not an option for correlation"
    alpha = 10**-1
    # create list of size distributions
    max_size = max(size_distribution.keys())
    size_dist = np.zeros(max_size-1)
    for size in sorted(size_distribution.keys()):
        size_dist[size-2] = size_distribution[size]

    # Use a set for hyperedges
    hyperedges = set()

    if first_N == 0:
        # Use all the nodes in every step
        nodes = list(range(N))
    else:
        # Limit the number of nodes to be selected for the largest hyperedges
        nodes = list(range(first_N))
    # Choose hyperedges at the largest size
    # ToDo: For now, just selecting them uniformly at random, might want to revisit later
    node_probabilities = np.array([1.0 for node in nodes])
    node_probabilities /= node_probabilities.sum()
    hyperedges, degrees = sample_hyperedges(nodes, max_size, size_distribution, node_probabilities)

    if first_N > 0:
        nodes = list(range(N))
        for node in nodes:
            if node not in degrees:
                degrees[node] = 0
        node_probabilities = np.array([1.0 for node in nodes])
        node_probabilities /= node_probabilities.sum()


    if correlation == "negative":
        node_probabilities = np.array([(1.0 / degrees[node])+alpha if
                                       degrees[node] > 0 else alpha for node in nodes])
        node_probabilities /= node_probabilities.sum()

    # for each smaller size
    for size in range(max_size-1, 1, -1):
        if size not in size_distribution.keys():
            continue

        if correlation in ["uncorrelated", "negative"]:
            # In the uncorrelated and negative cases, node_probabilities does not change
            sampled, degrees = sample_hyperedges(nodes, size, size_distribution, node_probabilities)
        elif correlation == "positive":
            # if positive, sample with prob of layer up
            node_probabilities = np.array([degrees[node]+alpha for node in nodes])
            node_probabilities /= node_probabilities.sum()
            sampled, degrees = sample_hyperedges(nodes, size, size_distribution, node_probabilities)
        elif correlation == "mixed":
            # if positive, sample with prob of layer up
            node_probabilities = np.array([(1.0 / degrees[node])+alpha if
                                           degrees[node] > 0 else alpha for node in nodes])
            node_probabilities /= node_probabilities.sum()
            sampled, degrees = sample_hyperedges(nodes, size, size_distribution, node_probabilities)

        # if mixed, sample with inverse prob of layer up
        hyperedges.update(sampled)

    return list(hyperedges)
