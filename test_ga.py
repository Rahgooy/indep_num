from ga import GA
import functions as FUN
import numpy as np
import lovasz as LOV
import algorithm_controller as ALG
import cvxopt
import time
#import extended_graph
from extended_graph import *
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
"""-----"""
def algorithm_controller_tests():
    n=20
    g =FUN.rand_graph(n,  n*(n-1)//3)
    while(g.independence_number()!=3):
        g =FUN.rand_graph(n,  n*(n-1)//3)
    indep = g.independence_number()
    print (FUN.fit(g))
    for i in range(10):
        print("now starting ", i)
        assert g.order()==n
        assert(g.independence_number()==indep)
        print (FUN.fit(g))
        print (len(g.maximal_independent_vertex_sets()))
        print (len(g.largest_independent_vertex_sets()))
        g = ALG.mutate_worst_vertex(g)

    print (FUN.fit(g))
    print (FUN._vertex_cost_list(g))
    print(str(g.adjacency_matrix()))

def incremental_test():
    ALG.incremental_ga()

"""-----"""
def run_tests():
    for i in range(20):
        helper_tests()
        crossover_tests()
        mutation_tests()
        fitness_tests()
    test_run_ga()
    test_large_lovasz_subgraph()
#run_tests()
#algorithm_controller_tests()
def test_lift_graph():
    g = FUN.rand_graph(10, 10*9//4)
    while(g.lovasz_theta()/g.independence_number() < 1.1):
        g = FUN.rand_graph(10, 10*9//4)

    for _ in range(5):
        g, _ = FUN.remove_extra_edges(g)
        # print(g.order())
        # print(g.lovasz_theta())
        # print(g.independence_number())
        lifts = [g.co_lift() for _ in range(50)]
        g = sorted(lifts, key = FUN.fit, reverse = True)[0]
        #print(g.order())
        print(FUN.fit(g))
        for i in range(1):
            g = FUN.large_lovasz_subgraph_vertex_count(g, 10)

def standard_expand_solution(B):
    BB = [row.tolist() + [0] for row in B]
    BB.append([0]*len(BB)+[1])
    return 0.75*np.array(BB)+0.25*np.identity(len(BB))

def fancy_expand_solution(B, neighbors):
    #epsilon = 5*10**-1
    epsilon = 0.5

    vectors = np.linalg.cholesky(B)
    handle = sum([v for v in vectors])
    # print (neighbors)
    # print("starting")
    #print(vectors)
    neighbor_vectors = [v for index, v in enumerate(vectors) if index in neighbors]
    to_neighbors = np.stack(neighbor_vectors).T
    from_neighbors = np.stack(neighbor_vectors)
    # M = from_neighbors @ np.vstack(handle)
    # print(len(neighbors))
    # print(np.vstack(handle))
    # print (from_neighbors)
    # print(M)
    # print(to_neighbors @ M)
    #
    # projection = (to_neighbors @ M).T[0]
    #
    # remainder = np.array(handle) - projection
    # if np.linalg.norm(remainder)<0.001:
    #     print("remainder too small")
    #     return standard_expand_solution(B)
    # else:
    #     remainder = 0.01*remainder/np.linalg.norm(remainder)
    #     vectors = np.stack(vectors.tolist() + [remainder])
    #     BB = vectors @ vectors.T
    #     return BB + 0.1*np.identity(len(BB))
    #
    # print(to_neighbors)
    #remainder = handle - sum()

    orth_neighbors, r = np.linalg.qr(to_neighbors)
    # print(np.vstack(handle))
    # print(orth_neighbors)
    # print("projection of handle")
    M = orth_neighbors @ orth_neighbors.T @ np.vstack(handle)
    print(M)
    print("wao")
    print(orth_neighbors @ orth_neighbors.T)
    extra_vector = np.vstack(handle) - M
    if np.linalg.norm(extra_vector) < 0.001:
        print("escaped")
        return standard_expand_solution(B)
    extra_vector = extra_vector/np.linalg.norm(extra_vector)
    # print(extra_vector.T[0])

    #vectors = [v if i not in neighbors else 0.99*v for i,v in enumerate(vectors)]
    #vectors.append(0.01*extra_vector.T[0])
    # print(vectors)

    BB = np.array(vectors) @ np.array(vectors).T
    print(len(BB))
    return BB + 0.1*np.identity(len(BB))
    #print(r)
    #proj_onto_neighbor_vectors =

    # print(neighbor_vectors)
    # print(np.linalg.norm(handle))
    # extra_vector = 0.5*handle
    #new_vectors = np.array((0.5*vectors).tolist() + [extra_vector])
    # print ("new vector")
    # print (len(vectors))
    # print (len(new_vectors))
    #BB = new_vectors @ new_vectors.T
    # print(np.linalg.det(BB))
    # print(BB)
    #
    # print(handle)
    # print (np.linalg.norm(handle))
    # BB = [row.tolist() + [0] for row in B]
    #
    #
    # BB.append([0]*len(BB)+[1])
    #
    #
    # I = [[1 if r ==c < len(BB)-1 else 0 for c in range(len(BB))] for r in range(len(BB))]
    # I = np.array(I)
    # BB= np.array(BB)
    # I = np.identity(len(BB))
    # eigen = np.linalg.eigh(BB)
    # from_eigenvectors = np.stack(eigen[1])
    # to_eigenvectors = from_eigenvectors.T
    # scaling_factors = [(epsilon - value) if value < epsilon else 0 for value in eigen[0]]
    # diagonal = np.diag(scaling_factors)
    # perturbation_matrix = from_eigenvectors @ diagonal @ to_eigenvectors
    # return BB + 0.5*perturbation_matrix

def test_lovasz_theta_initial_values(graph_size, iterations):
    g=[]
    solutions=[]
    dualstart = []
    B = []
    neighbors = []
    for i in range(iterations):
        g.append(FUN.rand_graph(10,10*9//4))
        solutions.append(lovasz_theta(g[i], long_return = True))
        #print (solutions[i]['theta'])
        g[i] = FUN.mutate_add_another_vertex(g[i])

    start_time=time.process_time()
    for i in range(iterations):
        x = lovasz_theta(g[i], long_return = True)
    end_time = time.process_time()
    without_hint_time = end_time- start_time
    for i in range(iterations):
        B.append(standard_expand_solution(solutions[i]['B']))
        dualstart.append({'zs':[cvxopt.matrix(B[i])]})

    start_time = time.process_time()
    for i in range(iterations):
        x = lovasz_theta(g[i], long_return = True, start = dualstart[i])
    end_time = time.process_time()
    standard_hint_time = end_time - start_time

    start_time = time.process_time()
    for i in range(iterations):
        neighbors.append(g[i].neighbors(g[i].order()-1))
        B[i] = fancy_expand_solution(solutions[i]['B'], neighbors[i])
        dualstart[i] = {'zs':[cvxopt.matrix(B[i])]}
    end_time = time.process_time()
    print("spent ", end_time-start_time, " just computing starting points.")
    start_time = time.process_time()
    for i in range(iterations):
        x = lovasz_theta(g[i], long_return = True, start = dualstart[i])
    end_time = time.process_time()
    fancy_hint_time = end_time - start_time

    return without_hint_time, standard_hint_time, fancy_hint_time

def test_evaluate_lovasz_theta_initial_values():
    print(test_lovasz_theta_initial_values(20, 1))
    #print(end_time-start_time)
    # g, _ = FUN.remove_extra_edges(g)
    # print(g.lovasz_theta())
    # print(g.independence_number())
    # g = g.complementer()
    # g.simplify()
    # lifted = g.random_lift()
    # lifted = lifted.complementer()
    # lifted.simplify()
    # lifted, _ = FUN.remove_extra_edges(lifted)
    # print(lifted.lovasz_theta())
    # print(lifted.independence_number())
    # back_down = FUN.large_lovasz_subgraph_vertex_count(lifted, 10)
    # back_down, _ = FUN.remove_extra_edges(back_down)
    # print(back_down.lovasz_theta())
    # print(back_down.independence_number())
    # back_down = back_down.complementer()
    # back_up = back_down.random_lift()
    # back_up = back_up.complementer()
    # back_up.simplify()
    # back_up, _ = FUN.remove_extra_edges(back_up)
    # print(back_up.lovasz_theta())
    # print(back_up.independence_number())
    #g = extended_graph.ExtendedGraph([(0,1),(1,2),(2,3),(3,4),(4,0)])
    #print(g.random_lift())
    #print(g.lift([(0,1)]))
#test_lift_graph()
#test_evaluate_lovasz_theta_initial_values()
#test_lovasz_theta_initial_values()

def troubleshooting_scratchpad():
    g = ExtendedGraph([(0,1),(1,2),(2,3),(1,3)])
    g.delete_edges([(0,1)])
    print(g.adjacency_matrix())
    permutation = list(range(g.order()))
    g.permute_vertices(permutation)
    print(g.adjacency_matrix())
troubleshooting_scratchpad()
