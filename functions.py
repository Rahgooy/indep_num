"""A library of fitness, mutation, and crossover functions."""
import sys
import numpy as np
import itertools
import bronkerbosch as BON
import random
import lovasz as LOV
from numpy.random import randint, rand
from extended_graph import *
from random import shuffle
<<<<<<< HEAD

"""Helper Functions"""
=======

"""Fitness Functions"""


def fit(g):
    if g.vcount() < 1:
        print("empty graph")
    return g.lovasz_theta() / g.independence_number()


def fit_eigen_values(g):
    """Returns the ratio between the largest and second largest abs. value eigenvectors."""
    """This doesn't give good results, because we usually must assume the graphs are regular."""
    adjacency = np.array(g.adjacency_matrix())
    eigenvalues = np.linalg.eigh(adjacency)[0]
    largest = eigenvalues[-1]
    second_largest = max(abs(eigenvalues[0]), abs(eigenvalues[-2]))
    return (largest - second_largest) / largest


def fit_regularity(g):
    """ returns the reciprocal of the standard deviation of the degree list """
    """ We take the reciprocal so that regular graphs are the most fit."""
    degrees = g.degree()
    deviation = np.std(degrees)
    return 1 / (1 + deviation)


def fit_with_regularity(g):
    """a weighted average of fitness and regularity."""
    return 0.95 * fit(g) + 0.05 * fit_regularity(g)


"""Mutation Functions"""


def mu(g):
    """Choose a random edge uv, if exists remove it if not add it"""
    g = g.copy()
    v = randint(0, g.vcount())
    u = randint(0, g.vcount())
    while u == v:
        u = randint(0, g.vcount())
    if g.has_edge(u, v):
        if g.size() > 1:
            g.delete_edges((u, v))
    else:
        g.add_edge(u, v)

    return g


def add_edge_to_max_indep_set(g):
    """Chooses a random maximal independent set to add an edge to"""
    g = g.copy()
    indep_sets = BON.cliques_of_graph(g.complementererer(), maximal=True)
    index = randint(0, len(indep_sets))
    indp = indep_sets[index]

    v = randint(0, len(indp))
    u = randint(0, len(indp))
    while u == v:
        u = randint(0, len(indp))
    g.add_edge(u, v)
    return g
>>>>>>> 1e8a37d83b63c85d2b09a9e2090f8433eb0684ae


def remove_extra_edges(g):
    """Calculates the maximal independent sets of g.
    If an edge doesnt intersect a maximal independent set, it can be removed
    without increasing the size of the independence number.
    We do this repeatedly until no such edges remain.
    """
    new_graph = g.copy()
    edges = len(new_graph.edges())
    indep_sets = None
    new_graph, indep_sets = _remove_extra_edge(new_graph, indep_sets)
    while (len(new_graph.edges()) != edges):
        edges = len(new_graph.edges())
        new_graph, indep_sets = _remove_extra_edge(new_graph, indep_sets)
    return new_graph, indep_sets


def _can_remove(e, max_indep_sets):
    """Returns true if we can remove this edge without affecting the independence number.
    If e[0] is in some max independent set, i, then i-{e[0]} U {e[1]} must be another max indep. set
    """
    sets_with_endpoint0 = [m for m in max_indep_sets if e[0] in m]
    for s in sets_with_endpoint0:
        if set([v for v in s if v != e[0]] + [e[1]]) in max_indep_sets:
            return False
    return True


def _update_indep_sets(g, e, indep_sets):
    """g is the new graph, with edge e removed.
    e is the edge which was removed,
    and indep_sets is a list of the maximal independent sets before the edge was removed.
    Returns the list of maximal independent sets of g.
    """
    non_neighbors_of_e = set([v for v in g.vertices() if not v in (g.neighbors(e[0]) + g.neighbors(e[1]))])
    subgraph_without_e = g.subgraph(non_neighbors_of_e)
    # new_indep_sets = BON.cliques_of_graph(subgraph_without_e.complementerer())
    new_indep_sets = [i.intersection(non_neighbors_of_e).union({e[0], e[1]}) for i in indep_sets]
    # [i for i in indep_sets if i not]
    extra_indep_sets = []
    for i in indep_sets:
        if not (e[0] in i) and (i.union({e[1]}) in new_indep_sets):
            if not (e[1] in i) and (i.union({e[0]}) in new_indep_sets):
                extra_indep_sets.append(i)
    new_indep_sets = new_indep_sets + extra_indep_sets
    return new_indep_sets


def _remove_extra_edge(g, indep_sets=None):
    """Returns a new graph by removing an edge from g. """
    # dict = BON.dict_from_get_adjacency(g.complementerer())
    # if indep_sets is None:
    #    indep_sets = BON.find_cliques(dict) #a list of all maximal-by-inclusion independent sets.
    indep_sets = BON.cliques_of_graph(g.complementerer())
    max_size = 0
    max_indep_sets = []  # a list of all maximal-by-size independent sets
    new_graph = g.copy()
    max_indep_sets = [i for i in indep_sets if len(i) == len(indep_sets[-1])]
    # removeable_edges = [e for e in g.edges() if _can_remove(e, max_indep_sets)]
    edges = g.edges()
    shuffle(edges)
    for e in edges:
        if _can_remove(e, max_indep_sets):
            new_graph.delete_edge(e)
            new_indep_sets = _update_indep_sets(new_graph, e, indep_sets)
            return new_graph, new_indep_sets
    return new_graph, indep_sets
    # vertices_in_max_indep_set = set(reduce(lambda x,y: union(x,y), max_indep_sets, set([])))
    if len(removeable_edges) == 0:
        # print "no edges to remove"
        return new_graph, indep_sets
    else:
        r = randint(0, len(removeable_edges) - 1)  # the -1 shouldn't be there, but it errors out without it.

        e = removeable_edges[r]
        # print "deleting ", e
        new_graph.delete_edge(e)
        # In the future, use update independent sets instead
        # new_indep_sets = BON.find_cliques((BON.dict_from_get_adjacency(new_graph.complementerer())))
        new_indep_sets = _update_indep_sets(new_graph, e, indep_sets)
        return new_graph, new_indep_sets

<<<<<<< HEAD
def _vertex_cost_list(g):
    """Returns a list of pairs [vertex_number,cost] sorted by cost."""
    solution = LOV.lovasz_theta(g, long_return = True)
    theta = solution['theta']
    witness = solution['B']
    costs = np.diagonal(witness)*theta
    costs = enumerate(costs) #adds an index
    costs = sorted(costs, key = lambda x: -x[1]) #sort by the cost
    return costs

def _large_lovasz_subgraph(g, fraction = 0.5):
    """Calculates lovasz theta of g, together with a witness.
    We use the costs of the vertices to identify a subgraph with a large lovasz theta.
    Then, we mutate one of the other edges."""
    theta = g.lovasz_theta()
    costs = _vertex_cost_list(g)
=======

def _large_lovasz_subgraph(g, fraction=0.5):
    """Calculates lovasz theta of g, together with a witness.
    We use the costs of the vertices to identify a subgraph with a large lovasz theta.
    Then, we mutate one of the other edges."""
    ans = LOV.lovasz_theta(g, long_return=True)
    theta = ans['theta']
    B = ans['B']
    costs = np.diagonal(B) * theta
    costs = enumerate(costs)  # adds an index
    costs = sorted(costs, key=lambda x: -x[1])  # sort by the cost
    # print costs
>>>>>>> 1e8a37d83b63c85d2b09a9e2090f8433eb0684ae
    valuable_vertices = []
    cur_sum = 0
    index = 0
    while (cur_sum < fraction * theta):
        valuable_vertices.append(costs[index][0])
        cur_sum += costs[index][1]
        index += 1
<<<<<<< HEAD
=======
    # values = [b**0.5 for b in diag]
>>>>>>> 1e8a37d83b63c85d2b09a9e2090f8433eb0684ae
    return valuable_vertices
"""Fitness Functions"""

def fit(g):
    if g.order() < 1:
        print("empty graph")
    return g.lovasz_theta() / len(g.independent_set())

def fit_regularity(g):
    """ returns the reciprocal of the standard deviation of the degree list """
    """ We take the reciprocal so that regular graphs are the most fit."""
    degrees = g.degree_sequence()
    deviation = np.std(degrees)
    return 1/(1+deviation)

def fit_with_regularity(g):
    """a weighted average of fitness and regularity."""
    return 0.90*fit(g) + 0.1*fit_regularity(g)

def fit_eigen_values(g):
    """Returns the ratio between the largest and second largest abs. value eigenvectors."""
    """This doesn't give good results, because we usually must assume the graphs are regular."""
    adjacency = np.array(g.adjacency_matrix())
    eigenvalues = np.linalg.eigh(adjacency)[0]
    largest = eigenvalues[-1]
    second_largest = max(abs(eigenvalues[0]),abs(eigenvalues[-2]))
    return (largest - second_largest) / largest
"""Mutation Functions"""
def mu(g):
    """Choose a random edge uv, if exists remove it. If not, add it"""
    g = g.copy()
    v = randint(0, g.order()-1)
    u = randint(0, g.order()-1)
    
    while u == v:
        u = randint(0, g.order()-1)
    if g.has_edge(u, v):
        if g.size() > 1:
            g.delete_edge(u, v)
    else:
        g.add_edge(u, v)
    # r = np.random.rand()
    # if r<0.01:
    #     g, _ = remove_extra_edges(g)
    return g

def add_edge_to_max_indep_set(g):
    """Chooses a random maximal independent set to add an edge to"""
    g = g.copy()
    indep_sets = BON.cliques_of_graph(g.complement(), maximal=True)
    index = randint(0,len(indep_sets)-1) #This causes an 'index out of range error.'
    indp = indep_sets[index]

    v = randint(0, len(indp))
    u = randint(0, len(indp))
    while u == v:
        u = randint(0, len(indp))
    g.add_edge(u,v)
    return g



def mutate_avoid_large_subgraph(g):
    """Finds the subgraph which contributes the most to theta.
    Adds a random edge which is not fully contained in that subgraph.
    """
    g = g.copy()
    valuable_vertices = _large_lovasz_subgraph(g, fraction=0.75)
    available_vertices = [v for v in g.vertices() if v not in valuable_vertices]
    u = np.random.choice(available_vertices)
    v = np.random.choice(g.vertices())
    while u == v:
        u = np.random.choice(available_vertices)
    if g.has_edge(u, v):
        if g.size() > 1:
            g.delete_edge(u, v)
    else:
        g.add_edge(u, v)

    return g

def mutate_add_then_remove_edges(g):
    """Adds edges randomly, then performs remove_extra_edges."""
    g = g.copy()
    g_c = g.complement()
    edges = g_c.edges()
    shuffle(edges)
    for e in edges[:15]:
        g.add_edge(e)
    g, _ = remove_extra_edges(g)
    return g
"""Crossover Functions"""


def cr1(g1, g2):
    """Create a new graph and add edges randomly from parents."""
    e1 = g1.edges()
    e2 = g2.edges()
    g = Graph({v: [] for v in range(0, g1.vcount())})
    m = (g1.size() + g2.size()) // 2

    i = 0
    while i < m:
        if rand() < 0.5:
            e = e1
        else:
            e = e2
        uv = e[randint(0, len(e))]
        g.add_edge(uv)
        i += 1
    return g


def cr2(g1, g2):
    """Create a new graph by randomly sampling the product of the parents uniformly."""
    # if not g.has_edge(uv):
    if g1.vcount() > 30 or g2.vcount() > 30:
        print("too large")
        return Graph({0: []})
    product = g1.disjunctive_product(g2)
    prob = 1.0 / (len(g1.independent_set()) * len(g2.independent_set()))
    sample = product.random_subgraph(prob)
    if sample.vcount() == 0:
        return Graph({0: []})
    return sample


def cr3(g1, g2, downsample=False):
    """Adds edges randomly between the disjoint union of the two graphs"""
    new_graph = g1.disjoint_union(g2, labels='pairs')
    print(new_graph.vertices())
    for a, b in itertools.product(g1.vertices(), g2.vertices()):
        r = np.random.rand()
        if r < 0.5:
            new_graph.add_edge(((0, a), (1, b)))
    if downsample:
        new_graph.random_subgraph(0.5, inplace=True)
    while new_graph.vcount() > 50:
        new_graph.random_subgraph(0.2, inplace=True)
    if new_graph.vcount() == 0:
        print
        "too small"
        return Graph({0: []})
    return new_graph


def cr4(g1, g2):
    """Keeps edges that are in both, flips a coin for edges that are in one but not the other."""
    new_graph = g1.copy()
    for edge in set(g1.edges()) ^ set(g2.edges()):
        r = np.random.rand()
        if r < 0.5:
            if new_graph.has_edge(edge):
                new_graph.delete_edge(edge)
            else:
                new_graph.add_edge(edge)
    # new_graph, _ = remove_extra_edges(new_graph)
    return new_graph


def cr5(g1, g2):
    """Flip a coin for each vertex. A pair of vertices whose smaller one is labeled g1
    is an edge iff g1 has that edge. """
    if g1.vcount() != g2.vcount():
        print("the two graphs should be of the same vcount")
        print(g1.vcount(), g2.vcount())
    vertex_assignments = np.random.randint(2, size=g1.vcount())
    new_graph = Graph.Full(g1.vcount())
    new_graph = ExtendedGraph.asExtended(new_graph.complementer())
    if new_graph.vcount() != g1.vcount():
        print("offf1111")
        print(new_graph.vcount(), g1.vcount())
    for v in new_graph.vertices():
        # print v
        if vertex_assignments[v] == 0:
            for k in [k for k in g1.neighbors(v) if k > v]:
                new_graph.add_edge(v, k)
        else:
            for k in [k for k in g2.neighbors(v) if k > v]:
                new_graph.add_edge(v, k)

<<<<<<< HEAD
    new_graph, _ = remove_extra_edges(new_graph)
    if new_graph.order()!=g1.order():
        print "grapsh have changed order."
        print new_graph.vertices()
    return new_graph

def cr6(g1, g2):
    """Orders the vertices of g1 and g2 according to their contribution to lovasz theta.
    Find subgraphs sg1 and sg2 such that sg1.order()+sg2.order()==g1.order()
                           and such that the total sum of contributions is maximized.
    add all edges between sg1 and sg2, then remove edges which don't affect the independence number.
    This function assumes g1 and g2 have the same number of vertices. Might fail otherwise.
    """
    costs_g1 = _vertex_cost_list(g1)
    costs_g2 = _vertex_cost_list(g2)
    index_g1 = 0
    index_g2 = 0
    while(index_g1 + index_g2 < g1.order()): #after this loop,
        if costs_g1[index_g1][1] > costs_g2[index_g2][1]:
            index_g1 += 1
        else:
            index_g2 += 1
    sg1 = g1.subgraph([c[0] for c in costs_g1[:index_g1]])
    sg2 = g2.subgraph([c[0] for c in costs_g2[:index_g2]])

    child_graph = sg1 + sg2 #These vertices are labeled [0..n]
    for v1, v2 in itertools.product(range(sg1.order()),range(sg2.order())):
        child_graph.add_edge(v1,v2+sg1.order())
    if child_graph.order()!= g1.order():
        print "order changed"
    child_graph, _ = remove_extra_edges(child_graph)
=======
    # new_graph, _ = remove_extra_edges(new_graph)
    if new_graph.vcount() != g1.vcount():
        print("grapsh have changed vcount.")
        print(new_graph.vertices())
    return new_graph

    # [np.random.rand() for v in g1.vertices()]
>>>>>>> 1e8a37d83b63c85d2b09a9e2090f8433eb0684ae

    return child_graph

def cr7(g1,g2):
    """Aligns the graphs according to vertex cost.
    When an edge is present in both graphs, we keep it.
    When it is only in one graph, we flip a coin.
    """
    costs_g1 = _vertex_cost_list(g1)
    g1_new_order = [c[0] for c in costs_g1] #list determines how to align the vertices of g1
    costs_g2 = _vertex_cost_list(g2)
    g2_new_order = [c[0] for c in costs_g2]
    #g2_new_order.reverse()
    dict={}
    for v in range(g1.order()):
        neighbors = []
        vertex_in_g1 = g1_new_order[v]
        vertex_in_g2 = g2_new_order[v]
        g1_neighbors = g1.neighbors(vertex_in_g1)
        g2_neighbors = g2.neighbors(vertex_in_g2)
        g1_neighbors = [a for a,b in enumerate(g1_new_order) if b in g1_neighbors]
        g2_neighbors = [a for a,b in enumerate(g2_new_order) if b in g2_neighbors]
        total_neighbors = g1_neighbors + g2_neighbors
        for t in total_neighbors:
            if t not in neighbors:
                if t in g1_neighbors and t in g2_neighbors:
                    neighbors.append(t)
                else:
                    r = np.random.rand()
                    if r>0.5:
                        neighbors.append(t)
        dict[v]=neighbors
    return Graph(dict)

def cr8(g1,g2):
    """ adds an edge if there is an edge in g1 or in g2"""
    new_graph = g1.copy()
    for e in g2.edges():
        new_graph.add_edge(e)
    new_graph,_=remove_extra_edges(new_graph)
    return new_graph

def rand_graph(n, m):
    "Generate a random graph with n vertices and m edges"
    g = ExtendedGraph()
    g.add_vertices(n)

    edges = []
    i = 0
    while i < m:
        x = randint(0, n)
        y = randint(0, n)
        if x > y:
            x, y = y, x
        if (x, y) not in edges:
            edges.append((x, y))
            i += 1
    g.add_edges(edges)
    return g
