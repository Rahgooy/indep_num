#!/usr/bin/python3
"""Contains methods coordinate runs of the genetic algorithm."""
from ga import GA
import functions as FUN
from numpy.random import randint, rand
from logger import *
from extended_graph import *
#from caching import print_cache_stats, reset_cache_number
from caching_redis import get_graphs_from_redis, set_graphs_to_redis

@wrap_with_log
def mutate_worst_vertex(g,mutation_options,choose_distinguished_vertex=False):
    """finds the lowest-cost vertex of g and removes it.
    Then adds another, distinguished vertex and optimizes its neighborhood."""
    # if choose_distinguished_vertex:
    #     subgraph_check = g.induced_subgraph(range(g.order()-1), implementation = "copy_and_delete").adjacency_matrix()
    g = g.copy()
    n=g.order()
    if choose_distinguished_vertex==True:
        permutation = list(range(n))
    g.permute_vertices(permutation)
    # if choose_distinguished_vertex:
    #     assert subgraph_check == g.induced_subgraph(range(g.order()-1), implementation = "copy_and_delete").adjacency_matrix()
    subgraph = g.induced_subgraph([w for w in g.vertices() if w!=n-1], implementation = "copy_and_delete")
    pop = [FUN.mutate_add_another_vertex(subgraph) for _ in range(mutation_options["pop_per_mu"])]
    #pop = pop+[g]
    ga = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr_distinguished, mutation_options["crossover_percent"], mutation_options["elite_percent"])
    results = ga.run(pop, mutation_options["iterations_per_mu"], mutation_options["branch_factor"])
    if len(results)==0:
        return g
    if mutation_options["branch_factor"] != 1:
        #good_results = sorted(results, key=FUN.fit, reverse=True)[:mutation_options["branch_factor"]]
        good_results = results[:mutation_options["branch_factor"]]
        good_graphs = good_results
        # for best in good_graphs:
        #     subgraph =  best.induced_subgraph(range(best.order()-1), implementation = "copy_and_delete").adjacency_matrix()
        #     assert subgraph == subgraph_check
        return good_graphs
    new_graph = results[0]
    best = new_graph
    return best
def add_vertex_and_mutate(g, mutation_options):
    #subgraph_check = g.induced_subgraph(range(g.order()), implementation = "copy_and_delete").adjacency_matrix()
    g = g.copy()
    #g=FUN.mutate_add_another_vertex(g)
    g.add_vertex()
    g=mutate_worst_vertex(g, mutation_options, choose_distinguished_vertex = True)
    # for item in g:
    #     subgraph = item.induced_subgraph(range(item.order()-1), implementation = "copy_and_delete").adjacency_matrix()
    #     assert subgraph_check == subgraph
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
    #independence_number =3
    #g = ExtendedGraph([(i,i+1) for i in range(4)] + [(4,0),(5,4),(5,2)])
    g = options["start_graph"]
    pop = [g]
    mutation_options = {"branch_factor":options["branch_factor"],"pop_per_mu":options["pop_per_mu"],
                        "iterations_per_mu":options["iterations_per_mu"], "elite_percent":options["elite_percent"], "crossover_percent":options["crossover_percent"]}
    genetic_alg1 = GA(FUN.fit, curry_add_vertex_and_mutate(mutation_options),
                          None, 0.0, options["meta_elite_percent"], pop_size = options["meta_pop"],make_unique=options["make_unique"])
    #pop = [g]
    results = genetic_alg1.run(pop, 31, meta_select_proc=True)
    print([FUN.fit(r) for r in results])
    #return sorted(results, key=FUN.fit, reverse = True)[0]
    return results[0]

def choose_level(start_graph):
    """Determines the level of the search tree best to examine."""
    top = 60 #the highest level we will examine
    n = start_graph.order()
    if rand() < 0.5:
        #choose the best level by binary search
        lower = n
        upper = top
        values = None
        while upper > lower+1 or values is None:
            level = (upper + lower) //2
            values = get_graphs_from_redis(level, start_graph)
            if values is None:
                upper = level
            else:
                lower = level
        return level, eval(values)

    else:
        #choose more or less randomly.
        level = randint(n,60)
        values = get_graphs_from_redis(level, start_graph)
        while values is None:
            level = (level+n)//2
            values = get_graphs_from_redis(level, start_graph)
    return level, eval(values)

def extend_search(mutation_options):
    """Picks one graph from the metapopulation and runs the G.A. for it."""
    level, values = choose_level(mutation_options["start_graph"])
    #values = eval(get_graphs_from_redis(level, mutation_options["start_graph"]))
    values.sort(key=lambda x: x[2])
    values[0][2]+=20
    set_graphs_to_redis(values)
    genetic_alg1 = GA(FUN.fit, curry_add_vertex_and_mutate(mutation_options),
                          None, 0.0, mutation_options["meta_elite_percent"], pop_size = mutation_options["meta_pop"],make_unique=mutation_options["make_unique"])
    pop = [values[0][0]]
    genetic_alg1.run(pop, 2, meta_select_proc =True)
