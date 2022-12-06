import numpy as np

"""
Define selection functions. For now should only accept the hypergraph,
but in the future may want to add a dictionary or function for computing
probabilities from external data (e.g. previous interaction) that may
not be stored in the hypergraph, or all data about interactions should
be stored as attributes
"""

"""
    Selects a hyperedge uniformly at random from H.
"""
def uniform_hyperedge(H):
    return np.random.choice(H.edges)

"""
    Selects an inactive hyperedge uniformly at random from H.
"""
def uniform_inactive(H):
    return np.random.choice(H.edges.filterby_attr("active", 0))

"""
    Selects an inactive hyperedge at random from H with probability
    proportional to hyperedge size.
"""
def biased_inactive(H):
    inactive_edges = H.edges.filterby_attr("active", 0)
    edge_sizes = np.array([float(len(H.edges.members(edge_id))) for edge_id in inactive_edges])
    edge_sizes /= edge_sizes.sum()
    if np.isnan(edge_sizes).any():
        print(edge_sizes)
    return np.random.choice(inactive_edges, p=edge_sizes)
