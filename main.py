#!/usr/bin/env sage -python

import sys
from ga import GA
from sage.all import *
import numpy as np
from numpy.random import randint, rand

def fit(g):
    if g.order() < 1:
        print("empty graph")
    return g.lovasz_theta() / (len(g.independent_set()) * g.order())

def mu(g):
    """Choose a random edge uv, if exists remove it if not add it"""
    g = g.copy()
    v = randint(0, g.order())
    u = randint(0, g.order())
    while u == v:
        u = randint(0, g.order())
    if g.has_edge(u, v):
        if g.size() > 1:
            g.delete_edge(u, v)
    else:
        g.add_edge(u, v)

    return g

def cr(g1, g2):
    """Create a new graph and add edges randomly from parents."""
    e1 = g1.edges()
    e2 = g2.edges()
    g = Graph({v:[] for v in range(0, g1.order())})
    m = (g1.size() + g2.size()) // 2

    i = 0
    while i < m:
        if rand() < 0.5:
            e = e1
        else:
            e = e2
        uv = e[randint(0, len(e))]
        if not g.has_edge(uv):
            g.add_edge(uv)
            i+=1
    return g



def rand_graph(n, m):
    "Generate a random graph with n vertices and m edges"
    g = { v: [] for v in range(n)}
    i = 0
    while i < m:
        x = randint(0, n)
        y = randint(0, n)
        if x > y:
            x, y = y, x
        if x != y and y not in g[x]:
            g[x].append(y)
            i += 1
    return Graph(g)

n = 7 # graph size
pop_size = 100
threshold = 0.11
pop = [rand_graph(10, randint(n, n*(n-1)/2 + 1)) for _ in range(pop_size)]

ga = GA(fit, mu, cr, 0.3, 0.2, cache_key=lambda x: str(x.adjacency_matrix()).__hash__())
results = ga.run(pop, 100, threshold)
results = sorted(results, key=lambda x: -x[1])
for g, fit in results:
    print(g.adjacency_matrix())
    print(g.lovasz_theta())
    print(len(g.independent_set()))
    r = g.lovasz_theta() / (len(g.independent_set()))
    print(r)
    print(fit)
    print("---------------------------------------")

hr = ga.cache_stats["hit"] * 100.0 / (ga.cache_stats["hit"] + ga.cache_stats["miss"])
print("Cache hit rate: " + str(round(hr)) + "%")
# G = rand_graph(5, 6)
# print(G.edges())
# G.add_edge(1, 5)
# print(G.edges())
# # G.plot().show()
# print G.lovasz_theta()
# print len(G.independent_set())
# print G.chromatic_number()
