#!/usr/bin/python3
"""Contains methods coordinate runs of the genetic algorithm.
"""
from ga import GA
import functions as FUN
from numpy.random import randint, rand
from logger import *
from caching import print_cache_stats


def mutate_worst_vertex(g, pop_size=100):
    """finds the lowest-cost vertex of g and removes it.
    Then adds another, distinguished vertex and optimizes its neighborhood."""
    v, cost = FUN._vertex_cost_list(g)[-1]  # the worst performing vertex
    print(cost)
    threshold = 1.2
    subgraph = g.induced_subgraph([w for w in g.vertices() if w != v], implementation="copy_and_delete")
    pop = [FUN.mutate_add_another_vertex(subgraph) for _ in range(pop_size)]
    ga = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2)
    results = ga.run(pop, 30, threshold)
    best = sorted(results, key=lambda x: -x[1])[0][0]
    # print(best)
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
    pop = [FUN.rand_graph(n, randint(n, n * (n - 1) / 2 + 1)) for _ in range(pop_size)]
    ga = GA(FUN.fit, FUN.mutate_add_then_remove_edges, FUN.cr6, 0.3, 0.2)
    results = ga.run(pop, 20, threshold)
    # logger.print_profile()

    # print(pop)
    while n < 30:
        print(n)
        best = sorted(results, key=lambda x: -x[1])[0]
        # print(best)
        best, _ = FUN.remove_extra_edges(best[0])
        pop = [best] * pop_size
        pop = [FUN.mutate_add_another_vertex(p) for p in pop]
        n += 1
        if n % 3 != 0:
            ga_inc = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2)
        else:
            ga_inc = GA(FUN.fit, FUN.mutate_composite, FUN.cr6, 0.5, 0.3)

        results = ga_inc.run(pop, 10, threshold)

        if n % 5 == 0:
            best = sorted(results, key=lambda x: -x[1])[0][0]
            for _ in range(5):
                best = mutate_worst_vertex(best)
            pop = [best] * pop_size
    print(FUN.fit(best))
    print(str(best.adjacency_matrix()))
    print(best.lovasz_theta())
    print(best.independence_number())

    results = sorted(results, key=lambda x: -x[1])
    with open("results_big.txt", "w") as f:
        for g, fit in results[:5]:
            f.write(str(g.adjacency_matrix()))
            f.write("\nTheta: " + str(g.lovasz_theta()))
            f.write("\nAlpha: " + str(g.independence_number()))
            r = g.lovasz_theta() / (g.independence_number())
            f.write("\nTheta/Alpha: " + str(r))
            f.write("\nFitness: " + str(fit))
            f.write("\n---------------------------------------\n")


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
        best = mutate_worst_vertex(best, 100)
    print(FUN.fit(best))
    print(str(best.adjacency_matrix()))
    print(best.lovasz_theta())
    print(best.independence_number())
