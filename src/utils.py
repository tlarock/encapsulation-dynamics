import pickle
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

"""
    Read a list of hyperedges, one per line with nodes
    separated by commas, from filename.
"""
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
    # Project the hyperedges into a network
    projection = project_hyperedges(hyperedges)
    # Get the list of connected components sorted from largest to smallest
    connected_components = sorted(list(nx.connected_components(projection)), key = lambda x: len(x), reverse=True)
    # We are only keeping nodes in the largest connected component
    nodes_to_keep = set(connected_components[0])
    connected_component = []
    for he in hyperedges:
        # Ignore single node hyperedges if called for
        if remove_single_nodes and len(he) == 1:
            continue

        # If the nodes are in the largest connected component
        if len(set(he).intersection(nodes_to_keep)) > 0:
            # Add hyperedge
            connected_component.append(he)

    return connected_component


"""
    Accepts results prefix of the form path/to/results/dataset/dataset and a
    dictionary of parameters, then reads the pickle file associated with that
    simulation if it exists.

    If given, can combine multiple random hypergraphs by giving a list of
    random_nums. By default uses results on hypergraph 0.
"""
def read_pickles(results_prefix, random_nums = [0], params_dict = dict()):


    obs_file = results_prefix + f"_{params_dict['selection']}_{params_dict['update']}_steps-{params_dict['steps']}_t-{params_dict['threshold']}_ia-{params_dict['ia']}"

    if "layer_randomization" in params_dict and params_dict["layer_randomization"]:
        rnd_template = results_prefix + "_layer_randomization" + f"_{params_dict['selection']}_{params_dict['update']}_steps-{params_dict['steps']}_t-{params_dict['threshold']}_ia-{params_dict['ia']}"
    else:
        rnd_template = results_prefix + "_randomization-{}" + f"_{params_dict['selection']}_{params_dict['update']}_steps-{params_dict['steps']}_t-{params_dict['threshold']}_ia-{params_dict['ia']}"



    obs_file += f"_runs-{params_dict['runs']}"
    rnd_template += f"_runs-{params_dict['runs']}"

    if "seeding_strategy" in params_dict:
        obs_file += f"_{params_dict['seeding_strategy']}"
        rnd_template += f"_{params_dict['seeding_strategy']}"

    # Check for seed function parameter
    if "seed_funct" in params_dict and params_dict["seed_funct"]:
        obs_file += "_" + params_dict["seed_funct"]
        rnd_template += "_" + params_dict["seed_funct"]

    if "drop_size" in params_dict and params_dict["drop_size"]:
        obs_file += f"_drop_size-{params_dict['drop_size']}"
        rnd_template += f"_drop_size-{params_dict['drop_size']}"

    obs_file += ".pickle"
    rnd_template += ".pickle"
    try:
        with open(obs_file, "rb") as fpickle:
            output_obs = pickle.load(fpickle)
    except Exception as e:
        print("Exception: " + str(e))
        output_obs = None

    try:
        if len(random_nums) == 1:
            with open(rnd_template.format(0), "rb") as fpickle:
                output_rnd = pickle.load(fpickle)
        else:
            output_rnd = aggregate_rand_pickles(rnd_template, random_nums)
    except Exception as e:
        print("Exception: " + str(e))
        output_rnd = None

    return output_obs, output_rnd


def aggregate_rand_pickles(template, random_nums):
    output_rnd = dict()
    for rnd in random_nums:
        with open(template.format(rnd), "rb") as fpickle:
            output = pickle.load(fpickle)
        for key in output:
            if key not in output_rnd:
                output_rnd[key] = output[key]
            else:
                output_rnd[key] = np.vstack((output_rnd[key], output[key]))
    return output_rnd

def drop_hyperedges_by_size(hyperedges, drop_size):
    return [he for he in hyperedges if len(he) != drop_size]

def remap_nodes(hyperedges):
    node_idx = 0
    node_mapping = dict()
    for he in hyperedges:
        for node in he:
            if node not in node_mapping:
                node_mapping[node] = node_idx
                node_idx += 1
    return [tuple([node_mapping[node] for node in he]) for he in hyperedges]
