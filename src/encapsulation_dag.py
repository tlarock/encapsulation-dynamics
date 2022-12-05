import math
import networkx as nx

def nodes_to_hyperedges(hyperedges):
    """
    Returns dictionary of nodes to hyperedges
    they participate in when hyperedges is a
    list-like of list-like edges.
    """

    nth = dict()
    he_map = dict()
    map_idx = 0
    for idx, he in enumerate(hyperedges):
        if he not in he_map:
            he_map[he] = map_idx
            map_idx += 1

        for node in he:
            if node not in nth:
                nth[node] = set()
            nth[node].add(he_map[he])
    return nth, he_map

def is_encapsulated(larger, smaller):
    return len(set(larger).intersection(set(smaller))) == len(smaller)

def encapsulation_dag(hyperedges):
    # Compute node to hyperedges
    nth, he_map = nodes_to_hyperedges(hyperedges)
    rev_map = {val:key for key,val in he_map.items()}
    # Construct the dag
    dag = nx.DiGraph()
    # Loop over hyperedges
    for he_idx, he in enumerate(hyperedges):
        dag.add_node(he)
        candidates = set()
        # Get all encapsulation candidate hyperedge ids
        for node in he:
            candidates.update([rev_map[i] for i in nth[node]])
        # for each candidate
        candidates_checked = set()
        for cand in candidates:
            if cand in candidates_checked:
                continue
            cand_idx = he_map[cand]
            if len(he) > len(cand):
                if is_encapsulated(he, cand):
                    dag.add_edge(he, cand)
            elif len(cand) > len(he):
                if is_encapsulated(cand, he):
                    dag.add_edge(cand, he)
    return dag, nth, he_map

def write_dag(dag, output_file):
    with open(output_file, 'w') as fout:
        nodes = list(dag.nodes())
        last_node = False
        for i in range(len(nodes)):
            if i == len(nodes)-1:
                last_node = True
            u = nodes[i]
            ne = list(dag.neighbors(u))
            if len(ne) > 0:
                for j, v in enumerate(ne):
                    if not last_node or j < len(ne)-1:
                        fout.write(','.join(map(str, u)) + "|" + ','.join(map(str, v)) + '\n')
                    else:
                        fout.write(','.join(map(str, u)) + "|" + ','.join(map(str, v)))
            else:
                if not last_node:
                    fout.write(','.join(map(str, u)) + '\n')
                else:
                    fout.write(','.join(map(str, u)))

def read_dag(input_file):
    dag = nx.DiGraph()
    with open(input_file, 'r') as fin:
        for line in fin:
            s = line.strip().split('|')
            if len(s) == 1:
                u = s[0].strip().split(',')
                if len(u) == 1:
                    u = tuple([int(u[0])])
                else:
                    u = tuple(map(int, u))
                dag.add_node(u)
            else:
                u,v = s
                u = tuple(map(int, u.strip().split(',')))
                v = tuple(map(int, v.strip().split(',')))
                dag.add_edge(u, v)
    return dag


def get_overlap_dists(dag, max_n=float('inf'), max_m=float('inf'),
                      binomial_norm=False, in_neighbors=False):
    """
    Compute the number of m-size hyperedges encapsulated
    by each n-size hyperedge and return the distribution
    for each (m,n) pair.
    """
    overlap = dict()
    he_overlap = dict()
    # For each hyperedge
    for he in dag.nodes():
        n = len(he)
        if n > max_n:
            continue

        if n not in overlap:
            overlap[n] = dict()

        # Get either in or out neighbors
        if not in_neighbors:
            ne = dag.successors(he)
        else:
            ne = dag.predecessors(he)

        # Loop over the hyperedges it encapsulates
        # and count how many of each size
        he_overlap = dict()
        for enc in ne:
            m = len(enc)
            if m > max_m:
                continue

            if m not in he_overlap:
                he_overlap[m] = 1
            else:
                he_overlap[m] += 1

        for m in he_overlap:
            if m not in overlap[n]:
                overlap[n][m] = []

            if not binomial_norm:
                overlap[n][m].append(he_overlap[m])
            elif not in_neighbors:
                overlap[n][m].append(he_overlap[m] / math.comb(n, m))
            else:
                overlap[n][m].append(he_overlap[m] / math.comb(dag.number_of_nodes()-n, m-n))


    return overlap
