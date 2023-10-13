import networkx as nx
class STrie:
    def __init__(self, hyperedges):
        """
        Accepts a list of hyperedges and constructs a set-trie and
        encapsulation DAG based on those hyperedges.

        Note: This implementation is preliminary with very limited testing
        (see main function below for examples). It is also implemented in
        a slightly convoluted way. It would probably make more sense to
        separately implement the GetAllSubsets function and call it before
        adding each hyperedge. Instead they are implemented together using
        the same stack and a flag for subset search. In the future one could
        also optimize (1) the ordering of the nodes, which is currently the
        results of sorted(); and (2) the subset search to simply use the
        DAG already constructed rather than searching every time.
        """
        # Initialize graphs for set-trie and dag
        T = nx.DiGraph()
        dag = nx.DiGraph()

        # Add the rootset to the set-trie
        rootset = (-1,)
        T.add_node(rootset)

        last_nodes = set()

        # Loop over hyperedges in order of increasing size
        for he in sorted(hyperedges, key = lambda l: len(l)):
            # Make sure that the hyperedge is sorted
            he_sort = sorted(he)
            he_strie = tuple(he_sort)

            # Initialize the stack to include all of the nodes in the current
            # hyperedge, with the first node coming last (first out).
            # All nodes except the first should have search=True, since they
            # should not be used to update the data structure, just to look
            # for existing subsets.
            stack = []
            for idx in range(1, len(he_sort)):
                stack.append((rootset, idx, tuple([he_sort[idx],]), True))
            idx = 0
            stack.append((rootset, idx, tuple([he_sort[idx],]), False))
            # Loop over all of the nodes
            while len(stack) > 0:
                # Pop an edge off of the stack
                prev, idx, curr, search = stack.pop()

                # If the current set is not yet a child
                # of prev, add it to the set-trie
                if not search and curr not in T[prev]:
                    T.add_edge(prev, curr)

                # Check if curr is in last_nodes
                if curr in last_nodes and curr != he_strie:
                    # If it is, it is definitely a DAG neighbor
                    dag.add_edge(he_strie, curr)

                # Loop over potential subsets of the current set
                if curr in T:
                    for sub in T[curr]:
                        if len(sub) < len(he_sort):
                            if len(set(he_strie).intersection(set(sub))) == len(sub):
                                if sub in last_nodes:
                                    dag.add_edge(he_strie, sub)
                                elif len(sub)+1 < len(he_sort):
                                    stack.append((curr, idx+1, sub, True))

                    # Grow the set by 1 element from the sorted hyperedge
                    idx += 1
                    if idx < len(he_sort):
                        new_set = tuple(sorted(curr + (he_sort[idx],)))
                        stack.append((curr, idx, new_set, False))
                    elif not search:
                        # If we have reached the end of this hyperedge,
                        # add it to the last_nodes list
                        if curr not in last_nodes:
                            last_nodes.add(curr)

        self.T = T
        self.dag = dag
        self.last_nodes = last_nodes


if __name__ == "__main__":
    """
    Two very simple tests implemented here.
    """

    hyperedges = [
        (1,3),
        (1,3,5),
        (1,4),
        (1,2,4),
        (1,3,4,5),
        (2,4),
        (2,3,5),
        (2,3,4,5)
    ]

    st = STrie(hyperedges)

    print(len(st.T.edges()))
    print(len(st.dag.edges()))
    print(len(hyperedges) == len(st.last_nodes))

    print(sorted(hyperedges, key=lambda l: len(l)))
    print(sorted(st.last_nodes, key=lambda l: len(l)))
    print(st.T.edges())
    print(st.dag.edges())

    hyperedges = [
        (1, 2, 3),
        (2, 3, 4, 5),
        (3, 4, 5)
    ]

    st = STrie(hyperedges)

    print(len(st.T.edges()))
    print(len(st.dag.edges()))
    print(len(hyperedges) == len(st.last_nodes))

    print(sorted(hyperedges, key=lambda l: len(l)))
    print(sorted(st.last_nodes, key=lambda l: len(l)))
    print(st.T.edges())
    print(st.dag.edges())
