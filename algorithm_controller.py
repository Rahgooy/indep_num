#!/usr/bin/python3
"""Contains methods coordinate runs of the genetic algorithm.
"""
from ga import GA
import functions as FUN
from numpy.random import randint, rand
from logger import *
from caching import print_cache_stats

@wrap_with_log
def mutate_worst_vertex(g,iterations,pop_size=100,return_size=1,choose_distinguished_vertex=False):
    """finds the lowest-cost vertex of g and removes it.
    Then adds another, distinguished vertex and optimizes its neighborhood."""
    g = g.copy()
    v = FUN.select_bad_vertex(g)
    fit = FUN.fit(g)
    n=g.order()
    permutation = [n-1 if x==v else v if x ==n-1 else x for x in range(n)]
    if choose_distinguished_vertex==True:
        permutation = list(range(n))
    g.permute_vertices(permutation)
    subgraph = g.induced_subgraph([w for w in g.vertices() if w!=n-1], implementation = "copy_and_delete")
    pop = [FUN.mutate_add_another_vertex(subgraph) for _ in range(pop_size-1)]
    pop = pop+[g]
    ga = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2)
    results = ga.run(pop, iterations, return_size)
    if len(results)==0:
        return g
    if return_size != 1:
        good_results = sorted(results, key=lambda x: -x[1])[:return_size]
        good_graphs = [FUN.remove_extra_edges(res[0])[0] for res in good_results]
        if FUN.fit(good_graphs[-1]) < fit:
            good_graphs[-1]= g
        return good_graphs
    new_graph, new_fit = sorted(results, key=lambda x: -x[1])[0]
    if fit < new_fit:
        best = new_graph
    else:
        best=g
    best, _ = FUN.remove_extra_edges(best)
    return best
def add_vertex_and_mutate(g):
    g = g.copy()
    g=FUN.mutate_add_another_vertex(g)
    g=mutate_worst_vertex(g, 10, return_size = platoon_size, choose_distinguished_vertex=True)
    return g

@wrap_with_log
def incremental_ga(initial_size, final_size,
                   iterations, iterations_between_updates,
                   pop_size = 100, independence_number=3):
    """Generates a random graph with a particular independence_number.

        mutate_worst_vertex {iterations_between_updates} times,

    """
    n = initial_size  # graph size
    pop_size = pop_size
    g =FUN.rand_graph(n,  n*(n-1)//3)
    while(g.independence_number()!=independence_number):
        g =FUN.rand_graph(n,  n*(n-1)//3)
    for _ in range(iterations_between_updates):
        g = mutate_worst_vertex(g, iterations, pop_size=pop_size)
    pop =[g]*pop_size

    while n < final_size:
        print(n)

        pop = [g]*pop_size
        pop = [FUN.mutate_add_another_vertex(p) for p in pop]
        n+=1
        ga_inc = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, 0.3, 0.2)

        results = ga_inc.run(pop, iterations)
        g = sorted(results, key=lambda x: -x[1])[0][0]
        for _ in range(iterations_between_updates):
            g = mutate_worst_vertex(g, iterations, pop_size=pop_size)
    print(FUN.fit(g))
    print(str(g.adjacency_matrix()))
    print(g.lovasz_theta())
    print(g.independence_number())

    results = sorted(results, key=lambda x: -x[1])

def incremental_with_mutate_worst_vertex():
    n = 20  # graph size
    pop_size = 100
    pop = [FUN.rand_graph(n, randint(n, n * (n - 1) / 2 + 1)) for _ in range(pop_size)]
    ga = GA(FUN.fit, FUN.mutate_add_then_remove_edges, FUN.cr6, 0.3, 0.2)
    results = ga.run(pop, 20)
    best = sorted(results, key=lambda x: -x[1])[0][0]
    print(best)
    best, _ = FUN.remove_extra_edges(best)
    for _ in range(20):
        best = mutate_worst_vertex(best,100)
    print(FUN.fit(best))
    print(str(best.adjacency_matrix()))
    print(best.lovasz_theta())
    print(best.independence_number())
@wrap_with_log
def advance_guard(vanguard, platoon_size, add_vertex=False):
    if not type(vanguard) is list:
        if not add_vertex:
            print("doh")
            return mutate_worst_vertex(vanguard, 10, return_size = platoon_size)
        else:
            g = add_vertex_and_mutate(g)
            return g
    return [advance_guard(troop, platoon_size, add_vertex = add_vertex) for troop in vanguard]

def evaluate_guard(vanguard):
    if not type(vanguard) is list:
        return FUN.fit(vanguard)
    else:
        return max([evaluate_guard(troop) for troop in vanguard])
@wrap_with_log
def retreat_guard(vanguard):
    return max(vanguard, key = evaluate_guard)

def depth(vanguard):
    if type(vanguard[0]) is list:
        return depth(vanguard[0]) + 1
    else:
        return 0
def best_soldier(vanguard):
     soldiers = [item for sublist in vanguard for item in sublist]
     return max(soldiers, key = FUN.fit)

def search_with_vanguard(shape=(5,2)):
    """maintains a list of extended_graphs.Each extended graph is mutated,
    and the best {shape[0]} mutants are kept. We maintain a tree structure of nested lists of graphs.
    This structure is called the 'vanguard,' the lowest-level lists are called 'platoons.'
    We 'retreat' the vanguard so that no more than {shape[1]} levels are present.
    This allows us to use a form of backtracking, which mitigates the loss of genetic diversity
    encountered when we mutate the vertices one-at-a-time."""
    n = 10  # graph size
    pop_size = 100
    independence_number =3
    g =FUN.rand_graph(n,  n*(n-1)//3)
    while(g.independence_number()!=independence_number):
        g =FUN.rand_graph(n,  n*(n-1)//3)
    # for _ in range(iterations_between_updates):
    #     g = mutate_worst_vertex(g, iterations, pop_size=pop_size, return_size = shape[0])
    #TODO: make add_vertex_and_mutate take a parameter, which is the number of graphs to return.
    #Accomodate the fact that this will be a list in GA.
    genetic_alg = GA(FUN.fit, add_vertex_and_mutate, FUN.cr_distinguished, 0.0, 0.2, pop_size = 5)
    pop = [g]
    results = genetic_alg.run(pop, iterations = 10, returnsize = 10)
    vanguard = [g]
    for _ in range(10):
        print("advance!")
        vanguard = advance_guard(vanguard, shape[0], add_vertex = True)
        if depth(vanguard)==shape[1]:
            vanguard = retreat_guard(vanguard)
    g = best_soldier(vanguard)
    return g
