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
    valuable_vertices = []
    cur_sum = 0
    index = 0
    while (cur_sum < fraction * theta):
        valuable_vertices.append(costs[index][0])
        cur_sum += costs[index][1]
        index += 1
    # values = [b**0.5 for b in diag]
    return valuable_vertices


def mutate_avoid_large_subgraph(g):
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

    # new_graph, _ = remove_extra_edges(new_graph)
    if new_graph.vcount() != g1.vcount():
        print("grapsh have changed vcount.")
        print(new_graph.vertices())
    return new_graph

    # [np.random.rand() for v in g1.vertices()]


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
