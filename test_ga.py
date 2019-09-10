from ga import GA
import functions as FUN
import numpy as np
import lovasz as LOV
import algorithm_controller as ALG
import cvxopt
import time
import redis
import interpret_results as INT
import igraph
#import extended_graph
from extended_graph import *
from start_graphs import *
from numpy.random import randint, rand
from caching_redis import get_graphs_from_redis


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
    g =random_gnp(50, .8)
    #indep_sets = BON.find_cliques(BON.dict_from_adjacency_matrix(g.complementer()))
    indep_sets = g.raw_maximal_independent_vertex_sets()
    #new_graph, new_indep_sets = FUN.remove_extra_edges(g,distinguished=True)
    #print(g.edges())
    start = time.process_time()
    new_indep_sets = FUN._update_indep_sets(g,g.edges()[-1],indep_sets)
    end = time.process_time()
    print("total time original: ", end-start)
    #correct_indep_sets = BON.find_cliques(BON.dict_from_adjacency_matrix(new_graph.complementer()))
    g.delete_edges(g.edges()[-1])
    new_graph = g.copy()
    correct_indep_sets = [set(i) for i in new_graph.maximal_independent_vertex_sets()]
    # print (new_indep_sets)
    # print (correct_indep_sets)
    for c in correct_indep_sets:
        assert c in new_indep_sets
    for i in new_indep_sets:
        to_assert = False
        for j in correct_indep_sets:
            if i.issubset(j):
                to_assert = True
        assert to_assert
#test_update_independent_sets()

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

def compute_starting_point(g):
    subgraph = g.induced_subgraph(range(g.order()-1))
    n_ratio = len(g.neighbors(g.order()-1))/(g.order()-1)
    subgraph.add_vertex()
    matrix1 = np.array(lovasz_theta(subgraph, long_return=True)['B'])
    subgraph = subgraph.induced_subgraph(range(subgraph.order()-1))
    subgraph = subgraph.complementer().simplify()
    subgraph.add_vertex()
    subgraph = subgraph.complementer().simplify()
    matrix2 = np.array(lovasz_theta(subgraph, long_return=True)['B'])
    return 0.75*((1-n_ratio)*matrix1+n_ratio*matrix2)+0.25*np.identity(g.order())


def test_lovasz_theta_initial_values(graph_size, iterations):
    g=[]
    gp=[]
    solutions=[]
    standard_solutions=[]
    dualstart = []
    B = []
    neighbors = []
    for i in range(iterations):
        g.append(FUN.rand_graph(graph_size,graph_size*(graph_size-1)//4))

        g[i].add_vertex()
        g[i].add_edges([(v,g[i].order()-1) for v in range(g[i].order()-1) if np.random.rand()<0.5])
    for i in range(iterations):
        subgraph = g[i].induced_subgraph(range(g[i].order()-1))
        standard_solutions.append(lovasz_theta(subgraph, long_return = True))
    start_time=time.process_time()
    for i in range(iterations):
        x = lovasz_theta(g[i], long_return = True)
    end_time = time.process_time()
    without_hint_time = end_time- start_time
    for i in range(iterations):
        #B.append(0.75*np.array(solutions[i]['B'])+0.25*np.identity(g[i].order()))
        dualstart.append({'zs':[cvxopt.matrix(compute_starting_point(g[i]))]})

    start_time = time.process_time()
    for i in range(iterations):
        x = lovasz_theta(g[i], long_return = True, start = dualstart[i])
    end_time = time.process_time()
    fancy_hint_time = end_time - start_time

    start_time = time.process_time()
    for i in range(iterations):
        B.append(standard_expand_solution(standard_solutions[i]['B']))
        dualstart[i] = {'zs':[cvxopt.matrix(B[i])]}
    end_time = time.process_time()
    #print("spent ", end_time-start_time, " just computing starting points.")
    start_time = time.process_time()
    for i in range(iterations):
        x = lovasz_theta(g[i], long_return = True, start = dualstart[i])
    end_time = time.process_time()
    standard_hint_time = end_time - start_time

    return without_hint_time, fancy_hint_time, standard_hint_time

def test_evaluate_lovasz_theta_initial_values():
    print(test_lovasz_theta_initial_values(20, 50))
#test_evaluate_lovasz_theta_initial_values()
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

# def troubleshooting_scratchpad():
#     g = ExtendedGraph([(0,1),(1,2),(2,3),(1,3)])
#     g.delete_edges([(0,1)])
#     print(g.adjacency_matrix())
#     permutation = list(range(g.order()))
#     g.permute_vertices(permutation)
#     print(g.adjacency_matrix())
#troubleshooting_scratchpad()
def redis_scratchpad():
    import socket
    print ("hooo")
    #print(socket.gethostname())
    r = redis.Redis(host="172.17.0.1", db=1)
    print(r.ping())

    start_time = time.process_time()
    x = 5
    end_time = time.process_time()
    #print (end_time-start_time)
    start_time = time.process_time()
    g = ExtendedGraph([(i,i+1) for i in range(4)]+[(4,0),(5,0)])
        #l = g.lovasz_theta()
    print(g.lovasz_theta())
    print(g.lovasz_theta())
    print(g.lovasz_theta())
    print(g.independence_number())
    #r.hset(str(g.adjacency_matrix()), "lovasz_theta",(5**0.5))
    #print (float((r.hget(str(g.adjacency_matrix()), "lovasz_theta"))))
    #print (g.lovasz_theta())
    print(g.calculate_start_matrix())
    print(g.calculate_start_matrix())
    for x in g.maximal_independent_vertex_sets():
        print(x)
    r.hset("tuna", "fish", "hongkong")# ["long","lost","love"])



    print(r.exists("tuna", "fish"))
    print(r.hkeys("tuna"))
    print(r.hget("tuna", "fish"))
    #print(r.hget("tuna", "fish"))
    end_time = time.process_time()
    #print( end_time - start_time)
#redis_scratchpad()

def test_calculate_indep_sets_from_subgraph():
    n=5
    g = FUN.rand_graph(n,n*(n-1)/4)
    g = FUN.mutate_add_another_vertex(g)
    g = FUN.mutate_add_another_vertex(g)
    g = FUN.mutate_add_another_vertex(g)
    g.add_vertex()
    g.add_edges([(vertex,g.order()-1) for vertex in range(g.order()-1)]  )
    #for vertex in range(g.order()-1):

    # for _ in range(25):
    #     g = FUN.mutate_add_another_vertex(g)
    subgraph = g.induced_subgraph(range(g.order()-1))
    #print(subgraph)
    #print(g.raw_maximal_independent_vertex_sets())
    #print(indep_sets_of_subgraph)
    start_time = time.process_time()
    g.maximal_independent_vertex_sets()
    end_time = time.process_time()
    print(end_time-start_time)
    start_time = time.process_time()
    g.maximal_independent_vertex_sets()
    end_time = time.process_time()
    print(end_time - start_time)

    start_time = time.process_time()
    g.raw_maximal_independent_vertex_sets()
    end_time = time.process_time()
    print(end_time - start_time)
    indep_sets_of_subgraph = subgraph.raw_maximal_independent_vertex_sets()
    start_time = time.process_time()
    indep_sets_of_g = FUN.calculate_independent_sets_from_subgraph(indep_sets_of_subgraph,g)
    end_time = time.process_time()
    print(end_time - start_time)
#test_calculate_indep_sets_from_subgraph()
def way_to_save_good_graphs():
    r = redis.Redis(host="172.17.0.1", db=1)
    n=5
    g = FUN.rand_graph(n,n*(n-1)/4)
    print(g.edges())
    red.lset()

#way_to_save_good_graphs()
def interpret_results():
    #g = INT.get_values_at_level(4)[0]
    #print(INT.check_state_independence(g[0]))
    INT.create_plot()
    # for i in range(7,10):
    #     values = get_graphs_at_level(i)
    #     print(values)
interpret_results()
def check_level_values():
    """Checks that the level is actually the number of vertices of the graph."""
    g = get_graphs_from_redis(4,start_graph_a3)
    print(g)
    g = get_graphs_from_redis(5,start_graph_a3)
    print(g)
    g = get_graphs_from_redis(6,start_graph_a3)
    print(g)
    g = get_graphs_from_redis(7,start_graph_a3)
    print(g)
#check_level_values()
def chromatic_number():
    n = 10
    g = ExtendedGraph.rand_graph(n,n*(n-1)/4)
    z = igraph.IGChromaticNumber(g)
    print(z)
#chromatic_number()
