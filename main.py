#!/usr/bin/python3
# from ga import GA
# import functions as FUN
# from numpy.random import randint, rand
# from logger import *
from algorithm_controller import *
"""
logger = Logger(sys.stdout, False)
n = 10  # graph size
pop_size = 100
threshold = 1.1
pop = [FUN.rand_graph(n, randint(n, n * (n - 1) / 2 + 1)) for _ in range(pop_size)]

ga = GA(FUN.fit_with_regularity, FUN.mutate_avoid_large_subgraph, FUN.cr5, 0.3, 0.2,
        cache_key=lambda x: str(x.adjacency_matrix()).__hash__(), log_func=logger)
"""
incremental_ga()

# G = rand_graph(5, 6)
# print(G.edges())
# G.add_edge(1, 5)
# print(G.edges())
# # G.plot().show()
# print G.lovasz_theta()
# print len(G.independent_set())
# print G.chromatic_number()
