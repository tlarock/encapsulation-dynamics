import numpy as np

def twonode_seed(H, configuration):
    num_seeds = configuration["initial_active"]
    activated_nodes_arr = np.random.choice([node for eid in H.edges
                             for node in H.edges.members(eid)
                             if len(H.edges.members(eid)) == 2], num_seeds)
    return activated_nodes_arr.tolist()
