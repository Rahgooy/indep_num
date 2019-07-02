from ga import GA
import functions as FUN
import numpy as np
import lovasz as LOV
from numpy.random import randint, rand


def test_crossover_function(l):
    """Expect l to be a crossover function.
    generates two random graphs and checks that l(g1, g2)
    does not error out and returns a graph of the same size."""
    g1 = FUN.random_gnp(5, .5)
    g2 = FUN.random_gnp(5, .5)
    child_graph = l(g1, g2)
    assert child_graph.order() == 5
    assert child_graph.is_simple()



def test_remove_extra_edges():
    """Checks that remove_extra_edges does not affect the independence number."""
    g = FUN.random_gnp(5, .5)
    r=g
    r, _ = FUN.remove_extra_edges(r)
    assert r.independence_number() == g.independence_number()
    assert r.order()==5

def test_update_independent_sets():
    """Generates a random graph, finds the independent sets,
    performs remove_extra_edges, and finds the independent sets again to
    ensure that remove_extra_edges returns the new independent sets correctly.
    """
    g = FUN.random_gnp(10, .5)
    #indep_sets = BON.find_cliques(BON.dict_from_adjacency_matrix(g.complementer()))
    new_graph, new_indep_sets = FUN.remove_extra_edges(g)
    #correct_indep_sets = BON.find_cliques(BON.dict_from_adjacency_matrix(new_graph.complementer()))
    correct_indep_sets = [set(i) for i in new_graph.maximal_independent_vertex_sets()]
    #print new_indep_sets
    #print correct_indep_sets
    for c in correct_indep_sets:
        assert c in new_indep_sets
    for i in new_indep_sets:
        assert i in correct_indep_sets

def test_add_edge_to_max_indep_set():
    g = FUN.random_gnp(10, .5)
    new_graph = FUN.add_edge_to_max_indep_set(g)
    print("test complete")

"""aggregate tests"""
def helper_tests():
    """runs all the helper tests"""
    test_update_independent_sets()
    test_remove_extra_edges()
def crossover_tests():
    """runs all the crossover tests"""
    crossovers = [FUN.cr4,FUN.cr5,FUN.cr6,FUN.cr7,FUN.cr8]
    #These are the crossover functions which preserve the order of the graph.
    for c in crossovers:
        #print(c.__name__)
        test_crossover_function(c)
def test_mutation_function(l):
    """expect l to be a mutation function."""
    g = FUN.random_gnp(5, .5)
    mutant_graph = l(g)
    print(l.__name__)
    #print mutant_graph.order()
    assert mutant_graph.order() == 5
    assert mutant_graph.is_simple()

def mutation_tests():
    mutation_functions = [FUN.mu, FUN.mutate_avoid_large_subgraph,FUN.mutate_add_then_remove_edges, FUN.add_edge_to_max_indep_set, FUN.mutate_distinguished_vertex]
    for m in mutation_functions:
        test_mutation_function(m)

def fitness_tests():
    g = FUN.random_gnp(10, .5)
    #print FUN.fit_with_regularity(g)
    return

"""aggregate tests"""

def test_run_ga():
    """Runs the genetic algorithm with various mutation and crossover functions to make
    sure that nothing errors out."""
    n = 5 # graph size
    pop_size = 100
    threshold = 1.130
    pop = [FUN.rand_graph(n, randint(n, n*(n-1)/2 + 1)) for _ in range(pop_size)]
    ga1 = GA(FUN.fit, FUN.mutate_add_then_remove_edges, FUN.cr6, 0.3, 0.2)
    results1 = ga1.run(pop, 20, threshold)
    pop = [FUN.rand_graph(n, randint(n, n*(n-1)/2 + 1)) for _ in range(pop_size)]
    ga2 = GA(FUN.fit_with_regularity, FUN.mu, FUN.cr7, 0.3, 0.2)
    results2 = ga2.run(pop, 20, threshold)
    ga3 = GA(FUN.fit, FUN.mutate_avoid_large_subgraph, FUN.cr5, 0.3, 0.2)
    results3 = ga3.run(pop, 20, threshold)
    pop = [FUN.rand_graph(n, randint(n, n*(n-1)/2 + 1))]*pop_size
    ga4 = GA(FUN.fit, FUN.mutate_distinguished_vertex, FUN.cr8, 0.3, 0.2)
    results4 = ga4.run(pop, 20, threshold)
def test_fit_regularity():
    g = FUN.random_gnp(5, .5)
    print(FUN.fit_regularity(g))

def test_large_lovasz_subgraph():

    g = FUN.random_gnp(10, .5)
    #FUN._subgraph_mutate(g)
    old_lov_theta = g.lovasz_theta()
    for i in range(10):
        FUN.mutate_avoid_large_subgraph(g)
    print("old theta: ", old_lov_theta)
    ans = LOV.lovasz_theta(g, long_return = True)
    theta = ans['theta']
    B = ans['B']
    print(theta, B)
    diag = np.diagonal(B)
    #values = [b**0.5 for b in diag]
    print(diag * theta)
    print(sum(diag*theta))
    assert abs(sum(diag*theta) - theta) < 0.01


def run_tests():
    for i in range(20):
        helper_tests()
        crossover_tests()
        mutation_tests()
        fitness_tests()
    test_run_ga()
    test_large_lovasz_subgraph()
run_tests()
