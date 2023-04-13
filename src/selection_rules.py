"""
Define selection functions. Functions defined here must accept the hypergraph
and lists representing the sizes of inactive edges and their indicies to
compute edge selection probabilities.
"""

import numpy as np

"""
    Selects an inactive hyperedge uniformly at random from H.
"""
def uniform_inactive(rng, H, inactive_edges_sizes,
                     sum_of_sizes,
                     inactive_edges_indices):
    index = rng.choice(inactive_edges_indices)
    return index

"""
    Selects an inactive hyperedge at random from H with probability
    proportional to hyperedge size.
"""
def biased_inactive(rng, H, inactive_edges_sizes,
                    sum_of_sizes,
                     inactive_edges_indices):
    inactive_edges_sizes /= sum_of_sizes
    index = rng.choice(inactive_edges_indices, p=inactive_edges_sizes)
    inactive_edges_sizes *= sum_of_sizes
    return index


"""
    Selects an inactive hyperedge at random from H with probability
    proportional to inverse of hyperedge size.
"""
def inverse_inactive(rng, H, inactive_edges_sizes,
                     sum_of_sizes,
                     inactive_edges_indices):
    # NOTE: Since I moved the handling of inverse calculations
    # inside of simulation.py, this function is now identical
    # to biased_inactive.
    inactive_edges_sizes /= sum_of_sizes
    index = rng.choice(inactive_edges_indices, p=inactive_edges_sizes)
    inactive_edges_sizes *= sum_of_sizes
    return index
