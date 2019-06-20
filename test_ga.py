#from sage.all import *
from ga import GA
import functions as FUN
import bronkerbosch as BON
#from main import cr3, cr4, fit_eigen_values
import numpy as np
import lovasz as LOV
from numpy.random import randint, rand

def fit(x):
    return 10 - (x[0] + x[1]) ** 2

def mu(x):
    new_x = x.copy()
    i = np.random.randint(len(x))
    new_x[i] = np.random.randint(-5, 5)
    return new_x

def cr(x1, x2):
    new_x = x1.copy()
    new_x[1] = x2[1]
    return new_x

g = GA(fit, mu, cr, 0.5, 0.1)
pop = [
        [5, 1],
        [3, 7],
        [12, 2],
        [1, 2],
        [13, 3],
        [1, 25],
        [-1, 7],
        [32, 34],
        [60, 50],
        [100, 1],
        [70, 21]
        ]
#r = g.run(pop, 10000, 8)
# #print(r)
# graph_list = [ graphs.BidiakisCube(),
#            graphs.ButterflyGraph(),
#            graphs.HeawoodGraph(),
#            graphs.HoffmanGraph(),
#            graphs.CubeGraph(4), #cospectral with above
#            graphs.DejterGraph(),
#            graphs.DyckGraph(),
#            graphs.GrotzschGraph(),
#            graphs.HoltGraph()]

def test_crossover_function(l):
    """Expect l to be a crossover function.
    generates two random graphs and checks that l(g1, g2)
    does not error out and returns a graph of the same size."""
    g1 = FUN.random_gnp(20, .5)
    g2 = FUN.random_gnp(20, .5)
    child_graph = l(g1, g2)
    assert child_graph.order() == 20
    assert child_graph.is_simple()
def test_cr3():
    """This test does not work."""
    for g1 in graph_list:
        for g2 in graph_list:
            q = FUN.cr3(g1, g2)
            assert q.order() == g1.order() + g2.order()

def test_cr4():
    """This function is idempotent"""
    for g in graph_list:
        q = FUN.cr4(g,g)
        assert q.is_isomorphic(g)

def test_eigen_fitness():
    """The complete graph should have eigenvalues [d, -1 -1 ... -1]
    where d is the degree."""
    k = graphs.CompleteGraph(15)
    value = FUN.fit_eigen_values(k)
    assert abs(value - 13/14.0) < 0.001
    g1 = graphs.PetersenGraph()
    g2 = graphs.ButterflyGraph()
    g = k + g2
    print(FUN.fit_eigen_values(g))

def test_remove_extra_edges():
    """Checks that remove_extra_edges does not affect the independence number."""
    g = FUN.random_gnp(20, .5)
    r=g
    r, _ = FUN.remove_extra_edges(r)
    #print r.independence_number()
    #print g.independence_number()
    assert r.independence_number() == g.independence_number()

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
    #test_cr4()
def test_mutation_function(l):
    """expect l to be a mutation function."""
    g = FUN.random_gnp(20, .5)
    mutant_graph = l(g)
    print(l.__name__)
    #print mutant_graph.order()
    assert mutant_graph.order() == 20
    assert mutant_graph.is_simple()

def mutation_tests():
    mutation_functions = [FUN.mu, FUN.mutate_avoid_large_subgraph,FUN.mutate_add_then_remove_edges, FUN.add_edge_to_max_indep_set]
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
    n = 10 # graph size
    pop_size = 100
    threshold = 1.130
    pop = [FUN.rand_graph(n, randint(n, n*(n-1)/2 + 1)) for _ in range(pop_size)]
    ga1 = GA(FUN.fit, FUN.mutate_add_then_remove_edges, FUN.cr6, 0.3, 0.2)
    results1 = ga1.run(pop, 20, threshold)
    ga2 = GA(FUN.fit_with_regularity, FUN.mu, FUN.cr7, 0.3, 0.2)
    results2 = ga2.run(pop, 20, threshold)
    ga3 = GA(FUN.fit, FUN.mutate_avoid_large_subgraph, FUN.cr5, 0.3, 0.2)
    results3 = ga3.run(pop, 20, threshold)

def run_tests():
    for i in range(1):
        helper_tests()
        crossover_tests()
        mutation_tests()
        fitness_tests()
    #test_run_ga()
run_tests()
#test_add_edge_to_max_indep_set()
#test_remove_extra_edges()
#run_tests()
def test_fit_regularity():
    g = FUN.random_gnp(10, .5)
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
#test_fit_regularity()
#test_remove_extra_edges()
#test_large_lovasz_subgraph()
