"""
Define selection functions. For now should only accept the hypergraph,
but in the future may want to add a dictionary or function for computing
probabilities from external data (e.g. previous interaction) that may
not be stored in the hypergraph, or all data about interactions should
be stored as attributes
"""

import numpy as np

"""
    Selects an inactive hyperedge uniformly at random from H.
"""
def uniform_inactive(H, inactive_edges_sizes,
                     inactive_edges_indices):
    index = np.random.choice(inactive_edges_indices)
    return index

"""
    Selects an inactive hyperedge at random from H with probability
    proportional to hyperedge size.
"""
def biased_inactive(H, inactive_edges_sizes,
                     inactive_edges_indices):

    edge_sizes = np.array(inactive_edges_sizes)
    edge_sizes /= edge_sizes.sum()
    index = np.random.choice(inactive_edges_indices, p=edge_sizes)
    return index


"""
    Selects an inactive hyperedge at random from H with probability
    proportional to inverse of hyperedge size.
"""
def inverse_inactive(H, inactive_edges_sizes,
                     inactive_edges_indices):
    edge_sizes = 1.0 / np.array(inactive_edges_sizes)
    edge_sizes /= edge_sizes.sum()
    index = np.random.choice(inactive_edges_indices, p=edge_sizes)
    return index

