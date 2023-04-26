import networkx as nx
import numpy as np

"""
Slightly modifying Phil's read data frunction for arbitrary paths to data in Austin Benson's format
"""
def read_data(path, t_min = None, t_max = None, multiedges=True):
    # read in the data
    nverts = np.array([int(f.rstrip('\n')) for f in open(path + 'nverts.txt')])
    times = np.array([float(f.rstrip('\n')) for f in open(path + 'times.txt')])
    simplices = np.array([int(f.rstrip('\n')) for f in open(path + 'simplices.txt')])

    times_ex = np.repeat(times, nverts)

    # time filtering

    t_ix = np.repeat(True, len(times))
    if t_min is not None:
        simplices = simplices[times_ex >= t_min]
        nverts = nverts[times >= t_min]
    if t_max is not None:
        simplices = simplices[times_ex <= t_max]
        nverts = nverts[times <= t_max]

    # relabel nodes: consecutive integers from 0 to n-1 

    unique = np.unique(simplices)
    mapper = {unique[i] : i for i in range(len(unique))}
    simplices = np.array([mapper[s] for s in simplices])

    # format as list of lists

    l = np.split(simplices, np.cumsum(nverts))
    C = [tuple(c) for c in l if len(c) > 0]
    if not multiedges:
        C_set = set([tuple(c) for c in C])
        C = [tuple(c) for c in C_set]

    return(C)

"""
Write a hypergraph as a list of hyperedges to a file.
"""
def write_hypergraph(hypergraph, output_file):
    with open(output_file, "w") as fout:
        for he in hypergraph:
            if len(he) > 0:
                fout.write(','.join(list(map(str, he))) + "\n")
            else:
                print(f"Found empty hyperedge. Ignoring.")
    return None

def read_hyperedges(filename):
    hyperedges = []
    with open(filename, "r") as fin:
        for line in fin:
            s = list(map(int, line.strip().split(',')))
            hyperedges.append(tuple(s))
    return hyperedges

"""
    Accept a list-like of hyperedges and construct
    an undirected graph based on co-occurrence.
"""
def project_hyperedges(hyperedges):
    G = nx.Graph()
    for he in hyperedges:
        for u in he:
            for v in he:
                if u != v:
                    G.add_edge(u,v)
    return G


"""
    Check the connectivity of the projection of
    a set of hyperedges by calling project_hyperedges
    then is_connected.
"""
def check_hyperedges_connectivity(hyperedges):
    return nx.is_connected(project_hyperedges(hyperedges))


"""
    Accepts a list of hyperedges as tuples. Computes the
    largest connected component of the hypergraph via its
    projection, then removes any hyperedges that are
    not connected.
"""
def largest_connected_component(hyperedges, remove_single_nodes=False):
    projection = project_hyperedges(hyperedges)
    connected_components = sorted(list(nx.connected_components(projection)), key = lambda x: len(x), reverse=True)
    nodes_to_remove = set([u for c in connected_components[1:] for u in c])
    connected_component = []
    for he in hyperedges:
        if len(set(he).intersection(nodes_to_remove)) == 0:
            if not remove_single_nodes or len(he) > 1:
                connected_component.append(he)

    return connected_component
