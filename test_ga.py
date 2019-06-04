from sage.all import *
from ga import GA
import functions as FUN
#from main import cr3, cr4, fit_eigen_values
import numpy as np

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
#print(r)
graph_list = [ graphs.BidiakisCube(),
           graphs.ButterflyGraph(),
           graphs.HeawoodGraph(),
           graphs.HoffmanGraph(),
           graphs.CubeGraph(4), #cospectral with above
           graphs.DejterGraph(),
           graphs.DyckGraph(),
           graphs.GrotzschGraph(),
           graphs.HoltGraph()]

def test_cr3():
    """The order should be additive"""
    for g1 in graph_list:
        for g2 in graph_list:
            q = FUN.cr3(g1, g2)
            print q.order()
            print g1.order()
            print g2.order()
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
    print FUN.fit_eigen_values(g)


def run_tests():
    #test_cr3()
    test_cr4()
    test_eigen_fitness()
run_tests()
