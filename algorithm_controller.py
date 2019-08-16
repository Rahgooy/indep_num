#!/usr/bin/python3
"""Contains methods coordinate runs of the genetic algorithm."""
from ga import GA
import functions as FUN
from numpy.random import randint, rand
from logger import *
from extended_graph import *
from caching import print_cache_stats, reset_cache_number

@wrap_with_log
def mutate_worst_vertex(g,mutation_options,choose_distinguished_vertex=False):
    """finds the lowest-cost vertex of g and removes it.
    Then adds another, distinguished vertex and optimizes its neighborhood."""
    if choose_distinguished_vertex:
        subgraph_check =  g.induced_subgraph(range(g.order()-1), implementation = "copy_and_delete").adjacency_matrix()


    g = g.copy()
    #v = FUN.select_bad_vertex(g)
    #fit = FUN.fit(g)
    n=g.order()
    #permutation = [n-1 if x==v else v if x ==n-1 else x for x in range(n)]
    if choose_distinguished_vertex==True:
        permutation = list(range(n))
    g.permute_vertices(permutation)
    if choose_distinguished_vertex:
        assert subgraph_check == g.induced_subgraph(range(g.order()-1), implementation = "copy_and_delete").adjacency_matrix()
        #assert g.adjacency_matrix()==subgraph_check
    subgraph = g.induced_subgraph([w for w in g.vertices() if w!=n-1], implementation = "copy_and_delete")
    pop = [FUN.mutate_add_another_vertex(subgraph) for _ in range(mutation_options["pop_per_mu"]-1)]
    pop = pop+[g]
    ga = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, mutation_options["crossover_percent"], mutation_options["elite_percent"])
    results = ga.run(pop, mutation_options["iterations_per_mu"], mutation_options["branch_factor"])
    if len(results)==0:
        return g
    if mutation_options["branch_factor"] != 1:
        #good_results = sorted(results, key=FUN.fit, reverse=True)[:mutation_options["branch_factor"]]
        good_results = results[:mutation_options["branch_factor"]]
        good_graphs = good_results
        #good_graphs = [FUN.remove_extra_edges(res)[0] for res in good_results]
        # if FUN.fit(good_graphs[-1]) < fit:
        #     good_graphs[-1]= g
        for best in good_graphs:
            subgraph =  best.induced_subgraph(range(best.order()-1), implementation = "copy_and_delete").adjacency_matrix()
            assert subgraph == subgraph_check
        return good_graphs
    #new_graph = sorted(results, key=FUN.fit, reverse=True)[0]
    new_graph = results[0]
    best = new_graph
    # if fit < FUN.fit(new_graph):
    #     best = new_graph
    # else:
    #     best=g
    #best, _ = FUN.remove_extra_edges(best)
    return best
def add_vertex_and_mutate(g, mutation_options):
    subgraph_check = g.induced_subgraph(range(g.order()), implementation = "copy_and_delete").adjacency_matrix()
    g = g.copy()
    g=FUN.mutate_add_another_vertex(g)
    g=mutate_worst_vertex(g, mutation_options, choose_distinguished_vertex = True)
    for item in g:
        subgraph = item.induced_subgraph(range(item.order()-1), implementation = "copy_and_delete").adjacency_matrix()
        assert subgraph_check == subgraph
    return g
def curry_add_vertex_and_mutate(mutation_options):
    """returns a function which applies add_vertex_and_mutate with a particular return size"""
    def to_return(g):
        return add_vertex_and_mutate(g,mutation_options)
    return to_return


@wrap_with_log
def search_with_vanguard(options):
    """maintains a list of extended_graphs. Each extended graph is mutated,
    and the best {shape[0]} mutants are kept. We maintain a tree structure of nested lists of graphs.
    This structure is called the 'vanguard,' the lowest-level lists are called 'platoons.'
    We 'retreat' the vanguard so that no more than {shape[1]} levels are present.
    This allows us to use a form of backtracking, which mitigates the loss of genetic diversity
    encountered when we mutate the vertices one-at-a-time.
    options = branch_factor, meta_pop, pop_per_mu, iterations_per_mu,
             elite_percent, crossover_percent, meta_elite_percent, make_unique,meta_select_proc
    """
    pop_size = options["meta_pop"]
    independence_number =3
    #g =FUN.rand_graph(n,  n*(n-1)//3)
    g = ExtendedGraph([(i,i+1) for i in range(4)] + [(4,0),(5,4),(5,2)])
    pop = [g]
    # for _ in range(options["meta_pop"]):
    #     while(g.independence_number()!=independence_number):
    #         print("stupid is as stupid does")
    #         g =FUN.rand_graph(n,  n*(n-1)//3)
    #     pop.append(g.copy())
    mutation_options = {"branch_factor":options["branch_factor"],"pop_per_mu":options["pop_per_mu"],
                        "iterations_per_mu":options["iterations_per_mu"], "elite_percent":options["elite_percent"], "crossover_percent":options["crossover_percent"]}
    genetic_alg1 = GA(FUN.fit, curry_add_vertex_and_mutate(mutation_options),
                          None, 0.0, options["meta_elite_percent"], pop_size = options["meta_pop"],make_unique=options["make_unique"])
    #pop = [g]
    results = genetic_alg1.run(pop, 15, meta_select_proc=options["meta_select_proc"])
    print([FUN.fit(r) for r in results])
    #return sorted(results, key=FUN.fit, reverse = True)[0]
    return results[0]
"""branch_factors = [1,2,5,7]
    meta_pops = [1,3,5,10,20]
    pop_per_mu = [50,100,200]
    iterations_per_mu = [5,10,15]
    elite_percent = [0.05]
    crossover_percent = [0,0.3]
    meta_elite_percent = [0.0, 0.2, 0.5]
    make_unique = [True, False]
    meta_select_proc = ["make_unique_then_select", "only_add_elites",
                        "make_extra_unique", "take_best", "take_best_unique"]
"""
