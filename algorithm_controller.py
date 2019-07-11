#!/usr/bin/python3
"""Contains methods coordinate runs of the genetic algorithm.
"""
from ga import GA
import functions as FUN
from numpy.random import randint, rand
from logger import *
from caching import print_cache_stats

def mutate_worst_vertex(g,logger=None,pop_size=50):
    """finds the lowest-cost vertex of g and removes it.
    Then adds another, distinguished vertex and optimizes its neighborhood."""
    g = g.copy()
    v = FUN.select_bad_vertex(g)
    fit = FUN.fit(g)
    if logger is None:
        logger = Logger(sys.stdout, False)
    threshold = 1.2
    n=g.order()
    permutation = [n-1 if x==v else v if x ==n-1 else x for x in range(n)]
    g.permute_vertices(permutation)
    subgraph = g.induced_subgraph([w for w in g.vertices() if w!=n-1], implementation = "copy_and_delete")
    pop = [FUN.mutate_add_another_vertex(subgraph) for _ in range(pop_size-1)]
    pop = pop+[g]
    ga = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2)
    results = ga.run(pop, 10, threshold)
    if len(results)==0:
        return g
    new_graph, new_fit = sorted(results, key=lambda x: -x[1])[0]
    if fit < new_fit:
        best = new_graph
    else:
        best=g
    best, _ = FUN.remove_extra_edges(best)
    return best

@wrap_with_log
def incremental_ga():
    """This function will need to be moved somewhere.
    runs the ga for
    """
    n = 10  # graph size
    pop_size = 100
    threshold = 1.1
    g =FUN.rand_graph(n,  n*(n-1)//3)
    while(g.independence_number()!=3):
        g =FUN.rand_graph(n,  n*(n-1)//3)
    for _ in range(2):
        g = mutate_worst_vertex(g)
    pop =[g]*pop_size

    print(FUN.fit(g))
    print(str(g.adjacency_matrix()))
    print(g.lovasz_theta())
    print(g.independence_number())
    #print(pop)
    while n < 13:
        print(n)

        pop = [g]*pop_size
        pop = [FUN.mutate_add_another_vertex(p) for p in pop]
        n+=1
        ga_inc = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2)

        results = ga_inc.run(pop, 20, threshold)
        g = sorted(results, key=lambda x: -x[1])[0][0]
        print (FUN.fit(g))
        print (g.independence_number())
        for _ in range(2):
            g = mutate_worst_vertex(g)
    print(FUN.fit(g))
    print(str(g.adjacency_matrix()))
    print(g.lovasz_theta())
    print(g.independence_number())

    results = sorted(results, key=lambda x: -x[1])
    global_logger.print_profile()

def incremental_with_mutate_worst_vertex():
    n = 20  # graph size
    pop_size = 100
    threshold = 1.2
    pop = [FUN.rand_graph(n, randint(n, n * (n - 1) / 2 + 1)) for _ in range(pop_size)]
    ga = GA(FUN.fit, FUN.mutate_add_then_remove_edges, FUN.cr6, 0.3, 0.2)
    results = ga.run(pop, 20, threshold)
    best = sorted(results, key=lambda x: -x[1])[0][0]
    print(best)
    best, _ = FUN.remove_extra_edges(best)
    for _ in range(20):
        best = mutate_worst_vertex(best,100)
    print(FUN.fit(best))
    print(str(best.adjacency_matrix()))
    print(best.lovasz_theta())
    print(best.independence_number())
