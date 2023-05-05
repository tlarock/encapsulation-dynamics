import sys
sys.path.append("../../src/")
from random_nested_hypergraph import *
from utils import check_hyperedges_connectivity

N = 10
max_size = 6
H = 2
epsilons = {i:1. for i in range(2, max_size)}
hyperedges = random_nested_hypergraph(N, max_size, H, epsilons)
while not check_hyperedges_connectivity(hyperedges):
    hyperedges = random_nested_hypergraph(N, max_size, H, epsilons)

# Map to consecutive integers
node_map = dict()
idx = 0
for he in hyperedges:
    for node in he:
        if node not in node_map:
            node_map[node] = idx
            idx += 1


hyperedges = [[node_map[u] for u in he] for he in hyperedges]
for he in sorted(hyperedges,
                 key=lambda k: len(k)):
    print(",".join(map(str, he)))
