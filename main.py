#!/usr/bin/env sage -python

import sys
from ga import GA
import functions as FUN
from numpy.random import randint, rand

n = 20 # graph size
pop_size = 100
threshold = 0.130
pop = [FUN.rand_graph(n, randint(n, n*(n-1)/2 + 1)) for _ in range(pop_size)]

ga = GA(FUN.fit, FUN.mutate_avoid_large_subgraph, FUN.cr5, 0.3, 0.2, cache_key=lambda x: str(x.adjacency_matrix()).__hash__())
results = ga.run(pop, 100, threshold)
results = sorted(results, key = lambda x: -x[1])
for g, fit in [results[0]]:
    print(g.adjacency_matrix())
    print(g.lovasz_theta())
    print(len(g.independent_set()))
    r = g.lovasz_theta() / (len(g.independent_set()))
    print(r)
    print(fit)
    print("---------------------------------------")

hr = ga.cache_stats["hit"] * 100.0 / (ga.cache_stats["hit"] + ga.cache_stats["miss"])
print("Cache hit rate: " + str(round(hr)) + "%")
# G = rand_graph(5, 6)
# print(G.edges())
# G.add_edge(1, 5)
# print(G.edges())
# # G.plot().show()
# print G.lovasz_theta()
# print len(G.independent_set())
# print G.chromatic_number()
