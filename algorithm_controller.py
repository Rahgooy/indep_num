#!/usr/bin/python3
"""Contains methods coordinate runs of the genetic algorithm.
"""
from ga import GA
import functions as FUN
from numpy.random import randint, rand
from logger import *
def mutate_worst_vertex(g,logger=None,pop_size=50):
    """finds the lowest-cost vertex of g and removes it.
    Then adds another, distinguished vertex and optimizes its neighborhood."""
    g = g.copy()
    #v, fit = FUN._vertex_cost_list(g)[-1] #the worst performing vertex
    v = FUN.select_bad_vertex(g)
    fit = FUN.fit(g)
    #print(cost)
    if logger is None:
        logger = Logger(sys.stdout, False)
    threshold = 1.2
    n=g.order()
    permutation = [n-1 if x==v else v if x ==n-1 else x for x in range(n)]
    g.permute_vertices(permutation)
    #pop = [g]*pop_size
    subgraph = g.induced_subgraph([w for w in g.vertices() if w!=n-1], implementation = "copy_and_delete")
    pop = [FUN.mutate_add_another_vertex(subgraph) for _ in range(pop_size-1)]
    pop = pop+[g]
    ga = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2,
            cache_key=lambda x: str(x.adjacency_matrix()).__hash__(), log_func=logger)
    results = ga.run(pop, 10, threshold)
    if len(results)==0:
        return g
    new_graph, new_fit = sorted(results, key=lambda x: -x[1])[0]
    if fit < new_fit:
        best = new_graph
    else:
        best=g
    #print(best)
    best, _ = FUN.remove_extra_edges(best)
    return best


def incremental_ga():
    """This function will need to be moved somewhere.
    runs the ga for
    """
    logger = Logger(sys.stdout, False)
    n = 10  # graph size
    pop_size = 100
    threshold = 1.1
    g =FUN.rand_graph(n,  n*(n-1)//3)
    while(g.independence_number()!=3):
        g =FUN.rand_graph(n,  n*(n-1)//3)
    for _ in range(2):
        g = mutate_worst_vertex(g)
    pop =[g]*pop_size
    # pop = [FUN.rand_graph(n, randint(n, n * (n - 1) / 2 + 1)) for _ in range(pop_size)]
    # ga = GA(FUN.fit, FUN.mutate_add_then_remove_edges, FUN.cr6, 0.3, 0.2,
    #         cache_key=lambda x: str(x.adjacency_matrix()).__hash__(), log_func=logger)
    # wrap_with_log(logger, ga, ['run', '_select', 'cr', 'mu', 'fit'])
    # results = ga.run(pop, 20, threshold)
    #logger.print_profile()


    #print(pop)
    while n < 50:
        print(n)
        #best = sorted(results, key=lambda x: -x[1])[0]
        #print(best)
        #best, _ = FUN.remove_extra_edges(best[0])
        pop = [g]*pop_size
        pop = [FUN.mutate_add_another_vertex(p) for p in pop]
        n+=1
        ga_inc = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2,
                      cache_key=lambda x: str(x.adjacency_matrix()).__hash__(), log_func=logger)
        # if n%3 != 0:
        #     ga_inc = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2,
        #             cache_key=lambda x: str(x.adjacency_matrix()).__hash__(), log_func=logger)
        # else:
        #     ga_inc = GA(FUN.fit, FUN.mutate_composite, FUN.cr6, 0.5, 0.3,
        #             cache_key=lambda x: str(x.adjacency_matrix()).__hash__(), log_func=logger)

        #wrap_with_log(logger, ga_inc, ['run', '_select', 'cr', 'mu', 'fit'])
        results = ga_inc.run(pop, 20, threshold)
        g = sorted(results, key=lambda x: -x[1])[0][0]
        print (FUN.fit(g))
        print (g.independence_number())
        for _ in range(2):
            g = mutate_worst_vertex(g, logger = logger)
    print(FUN.fit(g))
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
    logger.print_profile()

def incremental_with_mutate_worst_vertex():
    logger = Logger(sys.stdout, False)
    n = 20  # graph size
    pop_size = 100
    threshold = 1.2
    pop = [FUN.rand_graph(n, randint(n, n * (n - 1) / 2 + 1)) for _ in range(pop_size)]
    ga = GA(FUN.fit, FUN.mutate_add_then_remove_edges, FUN.cr6, 0.3, 0.2,
            cache_key=lambda x: str(x.adjacency_matrix()).__hash__(), log_func=logger)
    wrap_with_log(logger, ga, ['run', '_select', 'cr', 'mu', 'fit'])
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
