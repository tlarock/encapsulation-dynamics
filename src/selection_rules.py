"""
Define selection functions. Functions defined here must accept the hypergraph
and lists representing the sizes of inactive edges and their indicies to
compute edge selection probabilities.
"""

"""
    Selects an inactive hyperedge uniformly at random from H.
"""
def uniform_inactive(rng, H, inactive_edges_sizes,
                     inactive_edges_indices):
    index = rng.choice(inactive_edges_indices)
    return index

"""
    Selects an inactive hyperedge at random from H with probability
    proportional to hyperedge size.
"""
def biased_inactive(rng, H, inactive_edges_sizes,
                     inactive_edges_indices):
    edge_sizes = inactive_edges_sizes / inactive_edges_sizes.sum()
    index = rng.choice(inactive_edges_indices, p=edge_sizes)
    return index


"""
    Selects an inactive hyperedge at random from H with probability
    proportional to inverse of hyperedge size.
"""
def inverse_inactive(rng, H, inactive_edges_sizes,
                     inactive_edges_indices):
    edge_sizes = 1.0 / np.array(inactive_edges_sizes)
    edge_sizes /= edge_sizes.sum()
    index = rng.choice(inactive_edges_indices, p=edge_sizes)
    return index
