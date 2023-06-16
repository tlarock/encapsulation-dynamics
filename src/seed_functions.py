import numpy as np


### FUNCTIONS FOR NODE SEEDS ###
"""
    Randomly choose initial_active nodes from all nodes participating
    in 2-node edges.
"""
def twonode_seed(rng, H, configuration):
    initial_active = configuration["initial_active"]
    activated_nodes_arr = rng.choice(list(set([node for eid in H.edges
                             for node in H.edges.members(eid)
                             if len(H.edges.members(eid)) == 2])), replace=False, size = initial_active)
    return activated_nodes_arr.tolist()


"""
    Randomly choose initial_active nodes with probability proportional to the
    average size of the hyperedges the node participates in.
"""
def biased_seed(rng, H, configuration, inverse=False):
    initial_active = configuration["initial_active"]
    p = []
    for node in H.nodes:
        avg = np.mean([len(H.edges.members(eid)) for eid in H.nodes.memberships(node)])
        if inverse:
            avg = 1.0 / avg
        p.append(avg)

    p = np.array(p)
    p /= p.sum()
    activated_nodes_arr = rng.choice(list(H.nodes), p=p, replace=False, size=initial_active)
    return activated_nodes_arr.tolist()

"""
    Wrapper for inverse of the average hyperedge size bias.
"""
def inverse_biased_seed(rng, H, configuration):
    return biased_seed(rng, H, configuration, inverse=True)


"""
    Randomly choose initial_active nodes with probability proportional to
    the hyperdegree of the node (number of hyperedges the node
    participates in).
"""
def degree_biased_seed(rng, H, configuration, inverse=False):
    initial_active = configuration["initial_active"]
    p = np.array([float(H.nodes.degree[node]) for node in H.nodes])
    if inverse:
        p = 1.0 / p
    p /= p.sum()
    activated_nodes_arr = rng.choice(list(H.nodes), p=p, replace=False, size=initial_active)
    return activated_nodes_arr.tolist()

"""
    Wrapper for inverse of the hyperdegree.
"""
def inverse_degree_biased(rng, H, configuration):
    return degree_biased_seed(rng, H, configuration, inverse=True)

### FUNCTIONS FOR EDGE SEEDS ###
"""
    Randomly choose initial_active edges with probability proportional to the
    size of the edge.
"""
def size_biased_seed(rng, H, configuration, inverse=False):
    initial_active = configuration["initial_active"]
    p = np.array([float(len(H.edges.members(edge_id))) for edge_id in H.edges])
    if inverse:
        p = 1.0 / p
    p /= p.sum()
    activated_edges_arr = rng.choice(list(H.edges), p=p, replace=False, size=initial_active)
    return activated_edges_arr.tolist()

"""
    Wrapper for inverse of hyperedge size.
"""
def inverse_size_biased(rng, H, configuration):
    return size_biased_seed(rng, H, configuration, inverse=True)


"""
    Randomly choose 1 minimum-sized edge from each DAG connected component
    until initial_active is exhausted.

"""
def dag_components(rng, H, configuration):
    import networkx as nx
    dag = nx.DiGraph()
    # Compute the DAG and its components
    for edge_id in H.edges:
        dag.add_node(edge_id)
        for sup_id in H.edges[edge_id]["superfaces"]:
            dag.add_edge(sup_id, edge_id)

    components_lists = [list(c) for c in nx.weakly_connected_components(dag)]
    components_sizes = [np.array([1.0 / len(H.edges.members(edge_id)) for
                                  edge_id in c])
                        for c in components_lists]

    # Randomly choose a set of edges to activate by choosing 1 edge at a time
    # from each connected component, with a bias inversely proportional to the
    # size of the hyperedge
    activated_edges = set()
    while len(activated_edges) < configuration["initial_active"]:
        components_to_delete = set()
        # for each component
        for i in range(len(components_lists)):
            # Get the inverse edge sizes in component c
            inverse_sizes = components_sizes[i]

            # If all edge sizes are 0, we have activated all edges in this
            # component already, so we can safely skip it
            if sum(inverse_sizes) == 0:
                # If all of c's edges have already been added, skip
                components_to_delete.add(i)
                continue

            # Get the edges in the component as a list
            clist = components_lists[i]

            if len(components_lists) > 1:
                # Choose an edge that has not been chosen yet
                index = rng.choice(list(range(len(clist))), p = inverse_sizes / inverse_sizes.sum())

                # Add to activation set
                activated_edges.add(clist[index])

                # Set the size to 0 so it will not be picked again
                inverse_sizes[index] = 0.0
            else:
                # Just pick the rest of the edges from this component in one shot
                # Choose an edge that has not been chosen yet
                indices = rng.choice(list(range(len(clist))), p = inverse_sizes
                                     / inverse_sizes.sum(), replace=False,
                                     size=configuration["initial_active"]-len(activated_edges))
                for index in indices:
                    # Add to activation set
                    activated_edges.add(clist[index])

                    # Set the size to 0 so it will not be picked again
                    inverse_sizes[index] = 0.0

            if len(activated_edges) == configuration["initial_active"]:
                break

        components_lists = [c for i, c in enumerate(components_lists) if i not in components_to_delete]
        components_sizes = [c for i, c in enumerate(components_sizes) if i not in components_to_delete]

    return list(activated_edges)
