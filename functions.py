"""A library of fitness, mutation, and crossover functions."""
import sys
import numpy as np
import itertools
import random
import lovasz as LOV
from numpy.random import randint, rand
from extended_graph import *
from random import shuffle
from caching import wrap_extended_graph_method_with_cache as wrap_with_cache
from logger import wrap_with_log

"""Helper Functions"""

@wrap_with_log
def edge_list_from_dict(dict):
    edge_list = []
    for k in dict.keys():
        for v in dict[k]:
            if k < v:
                edge_list.append((k, v))
    return edge_list

@wrap_with_log
def rand_graph(n, m):
    """Generate a random graph with n vertices and m edges"""
    g = {v: [] for v in range(n)}
    i = 0
    while i < m:
        x = randint(0, n)
        y = randint(0, n)
        if x > y:
            x, y = y, x
        if x != y and y not in g[x]:
            g[x].append(y)
            i += 1
    r_graph = ExtendedGraph(edge_list_from_dict(g))
    while r_graph.order() < n:
        r_graph.add_vertex()
    return r_graph


@wrap_with_log
def random_gnp(n, p):
    """Generate a random graph where each edge has probability p"""
    dict = {}
    for a in range(n):
        neighbors = []
        for b in range(a + 1, n):
            r = np.random.rand()
            if r < p:
                neighbors.append(b)
        dict[a] = neighbors
    g = ExtendedGraph(edge_list_from_dict(dict))
    while (g.order() < n):  # this can occur if the random graph is not connected
        g.add_vertex()
    assert g.order() == n
    return g


@wrap_with_log
def remove_extra_edges(g, distinguished=False):
    assert distinguished
    """Calculates the maximal independent sets of g.
    If an edge doesnt intersect a maximal independent set, it can be removed
    without increasing the size of the independence number.
    We do this repeatedly until no such edges remain.
    If distinguished = True, then we only remove the edges with the last vertex
    """
    order = g.order()
    new_graph = g.copy()
    edges = len(new_graph.edges())
    indep_sets = None
    new_graph, indep_sets = _remove_extra_edge(new_graph, indep_sets, distinguished=distinguished)
    while (len(new_graph.edges()) != edges):
        edges = len(new_graph.edges())
        new_graph, indep_sets = _remove_extra_edge(new_graph, indep_sets, distinguished = distinguished)
    assert (new_graph.order() == order)
    return new_graph, indep_sets


@wrap_with_log
def _can_remove(e, max_indep_sets):
    """Returns true if we can remove this edge without affecting the independence number.
    If e[0] is in some max independent set, i, then i-{e[0]} U {e[1]} must be another max indep. set
    """
    sets_with_endpoint0 = [m for m in max_indep_sets if e[0] in m]
    for s in sets_with_endpoint0:
        if set([v for v in s if v != e[0]] + [e[1]]) in max_indep_sets:
            return False
    return True


@wrap_with_log
def _update_indep_sets(g, e, indep_sets):
    """g is the new graph, with edge e removed.
    e is the edge which was removed,
    and indep_sets is a list of the maximal independent sets before the edge was removed.
    Returns the list of maximal independent sets of g.
    """
    non_neighbors_of_e = set([v for v in g.vertices() if not v in (g.neighbors(e[0]) + g.neighbors(e[1]))])
    if len(non_neighbors_of_e)==0:
        new_indep_sets = [ [e[0],e[1]] ]
        return indep_sets + new_indep_sets
    subgraph_without_e = g.subgraph(non_neighbors_of_e)
    indep_sets_of_subgraph = [set(i).intersection(subgraph_without_e.vertices()) for i in indep_sets]
    s=[]
    for i in indep_sets_of_subgraph:
        s.append(i)
    indep_sets_of_subgraph = s #made indep_sets_of_subgraph unique
    new_indep_sets = [list(i) + [e[0],e[1]] for i in indep_sets_of_subgraph]
    return indep_sets + new_indep_sets



@wrap_with_log
def _remove_extra_edge(g, indep_sets=None, distinguished=False):
    assert distinguished
    if distinguished:
        subgraph_check = g.induced_subgraph(range(g.order()-1), implementation="copy_and_delete").adjacency_matrix()

    order = g.order()
    """Returns a new graph by removing an edge from g. """
    if indep_sets is None:
        indep_sets = [set(i) for i in g.maximal_independent_vertex_sets()]
    max_size = 0
    new_graph = g.copy()
    max_indep_sets = [i for i in indep_sets if len(i) == len(indep_sets[-1])]
    if not distinguished:
        edges = g.edges()
    else:
        edges = [e for e in g.edges() if g.vertices()[-1] in e]
    shuffle(edges)
    for e in edges:
        if _can_remove(e, max_indep_sets):
            new_graph.delete_edges(e)
            new_indep_sets = _update_indep_sets(new_graph, e, indep_sets)
            if distinguished:
                assert new_graph.induced_subgraph(range(new_graph.order()-1),implementation="copy_and_delete").adjacency_matrix()==subgraph_check
            return new_graph, new_indep_sets
    assert (order == g.order())
    if distinguished:
        assert g.induced_subgraph(range(g.order()-1),implementation="copy_and_delete").adjacency_matrix()==subgraph_check
    return new_graph, indep_sets





@wrap_with_log
def _large_lovasz_subgraph(g, fraction=0.5):
    """Calculates lovasz theta of g, together with a witness.
    We use the costs of the vertices to identify a subgraph with a large lovasz theta.
    Then, we mutate one of the other edges."""
    theta = g.lovasz_theta()
    costs = g.vertex_cost_list()
    valuable_vertices = []
    cur_sum = 0
    index = 0
    while (cur_sum < fraction * theta):
        valuable_vertices.append(costs[index][0])
        cur_sum += costs[index][1]
        index += 1
    return valuable_vertices

def large_lovasz_subgraph_vertex_count(g, count):
    """Returns the subgraph of g with {count} vertices such that the subgraph has largest value"""
    #return g.subgraph(random.sample(g.vertices(), count))
    costs = g.vertex_cost_list()
    n=g.order()
    valuable_vertices = []
    i = 0
    while i < count:
        valuable_vertices.append(costs[i][0])
        i += 1
    return g.subgraph(valuable_vertices)

@wrap_with_log
def select_bad_vertex(g):
    """Uses a roulette selection to find a vertex which contributes little to lovasz theta.
       Returns the bad vertex and its cost.
    """
    costs = g.vertex_cost_list()
    values = [1/(c[1]**2+0.001) for c in costs]
    cdf = np.cumsum(values)
    cdf = cdf / cdf[-1]
    r = np.random.rand()
    sample = sum(r > cdf)
    return(costs[sample][0])

"""Fitness Functions"""


@wrap_with_log
def fit(g):
    order = g.order()
    value = g.lovasz_theta() / g.independence_number()
    assert (g.order() == order)
    if value > 1.5:
        print (g.adjacency_matrix())
    return value


@wrap_with_log
@wrap_with_cache
def fit_regularity(g):
    """ returns the reciprocal of the standard deviation of the degree list """
    """ We take the reciprocal so that regular graphs are the most fit."""
    degrees = g.degree()
    deviation = np.std(degrees)
    return 1 / (1 + deviation)


@wrap_with_log
@wrap_with_cache
def fit_with_regularity(g):
    """a weighted average of fitness and regularity."""
    return 0.90 * fit(g) + 0.1 * fit_regularity(g)


@wrap_with_log
@wrap_with_cache
def fit_eigen_values(g):
    """Returns the ratio between the largest and second largest abs. value eigenvectors."""
    """This doesn't give good results, because we usually must assume the graphs are regular."""
    adjacency = np.array(g.adjacency_matrix())
    eigenvalues = np.linalg.eigh(adjacency)[0]
    largest = eigenvalues[-1]
    second_largest = max(abs(eigenvalues[0]), abs(eigenvalues[-2]))
    return (largest - second_largest) / largest


"""Mutation Functions"""


@wrap_with_log
def mu(g):
    """Choose a random edge uv, if exists remove it. If not, add it"""
    order = g.order()
    g = g.copy()
    v = randint(0, g.order())
    u = randint(0, g.order())

    while u == v:
        u = randint(0, g.order())
    if g.has_edge(u, v):
        if g.size() > 1:
            g.delete_edges([(u, v)])
    else:
        g.add_edges([(u, v)])
    assert (order == g.order())
    return g


@wrap_with_log
def add_edge_to_max_indep_set(g):
    """Chooses a random maximal independent set to add an edge to"""
    order = g.order()
    g = g.copy()
    # indep_sets = BON.cliques_of_graph(g.complement(), maximal=True)
    indep_sets = g.largest_independent_vertex_sets()
    index = randint(0, len(indep_sets))
    indp = indep_sets[index]

    v = randint(0, len(indp))
    u = randint(0, len(indp))
    while u == v:
        u = randint(0, len(indp))
    g.add_edges([(indp[u], indp[v])])
    assert (g.order() == order)
    return g


@wrap_with_log
def mutate_avoid_large_subgraph(g):
    order = g.order()
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
            g.delete_edges([(u, v)])
    else:
        g.add_edge(u, v)
    assert (order == g.order())
    return g


@wrap_with_log
def mutate_remove_then_remove_edges(g):
    """removes one edge, then performs remove extra_edges."""
    order = g.order()
    g = g.copy()
    e = g.edges()[randint(0, len(g.edges()))]
    g.delete_edges([e])
    g, _ = remove_extra_edges(g)
    assert (g.order() == order)
    return g


@wrap_with_log
def mutate_add_then_remove_edges(g):
    """Adds edges randomly, then performs remove_extra_edges."""
    order = g.order()
    g = g.copy()
    g_c = g.complementer().simplify()
    edges = g_c.edges()
    shuffle(edges)
    g.add_edges(edges[:(3 * order // 2)])
    g, _ = remove_extra_edges(g)
    assert (g.order() == order)
    return g


@wrap_with_log
def mutate_composite(g):
    """Either uses mutate_add_then_remove_edges or mutate_remove_then_remove_edges"""
    r = np.random.rand()
    if r < 0.1:
        return mutate_remove_then_remove_edges(g)
    else:
        return mutate_add_then_remove_edges(g)


@wrap_with_log
def mutate_distinguished_vertex(g):
    subgraph = g.induced_subgraph(range(g.order()-1), implementation = "copy_and_delete")
    subgraph_check =  subgraph.adjacency_matrix()
    order = g.order()
    subgraph_order = order -1
    subgraph_edge_density = subgraph.ecount()/(order*(order-1)/2)
    """Assumes the last vertex of g was just added.
    adds an edge between the distinguished vertex and other vertices with probability 0.2
    Then removes unnecessary edges."""
    distinguished_vertex = g.vertices()[-1]
    for v in range(g.order() - 1):
        r = np.random.rand()
        if r < subgraph_edge_density:
            g.add_edge(v, distinguished_vertex)
    g.simplify()
    assert subgraph_check ==  g.induced_subgraph(range(g.order()-1), implementation = "copy_and_delete").adjacency_matrix()

    g, _ = remove_extra_edges(g, distinguished=True)
    # other_vertex = randint(0,g.order()-1)
    # if g.has_edge(other_vertex, distinguished_vertex):
    #     g.delete_edges([(other_vertex, distinguished_vertex)])
    # else:
    #     g.add_edge(other_vertex, distinguished_vertex)
    assert (g.order() == order)
    assert subgraph_check ==  g.induced_subgraph(range(g.order()-1), implementation = "copy_and_delete").adjacency_matrix()
    return g


@wrap_with_log
def mutate_add_another_vertex(g):
    """Adds another vertex to the graph.
    Chooses a vertex from each maximal independent set as a neighbor to the new vertex.
    """
    new_graph = g.copy()
    # vertex_assignments = np.random.randint(2, size=g.order())
    indep_sets = g.largest_independent_vertex_sets()
    neighbors = [a[randint(0, len(a))] for a in indep_sets]
    new_graph.add_vertex()
    new_graph.add_edges([(n, new_graph.vertices()[-1]) for n in neighbors])
    new_graph = new_graph.simplify()

    M= new_graph.induced_subgraph(range(new_graph.order()-1), implementation= "copy_and_delete").adjacency_matrix()
    assert M == g.adjacency_matrix()
    # for v in g.vertices():
    #     if v!= g.order()-1 and vertex_assignments[v] ==1:
    #         new_graph.add_edge(v, new_graph.vertices()[-1])
    return new_graph


"""Crossover Functions"""


@wrap_with_log
def cr4(g1, g2):
    """Keeps edges that are in both, flips a coin for edges that are in one but not the other."""
    new_graph = g1.copy()
    for edge in set(g1.edges()) ^ set(g2.edges()):
        r = np.random.rand()
        if r < 0.25:
            if new_graph.has_edge(edge[0], edge[1]):
                new_graph.delete_edges(edge)
            else:
                new_graph.add_edges([edge])
    # new_graph, _ = remove_extra_edges(new_graph)
    return new_graph


@wrap_with_log
def cr5(g1, g2):
    """Flip a coin for each vertex. A pair of vertices whose smaller one is labeled g1
    is an edge iff g1 has that edge. """
    assert (g1.order() == g2.order())
    vertex_assignments = np.random.randint(2, size=g1.order())
    # new_graph = graphs.CompleteGraph(g1.order()).complement()
    new_graph = ExtendedGraph([])
    new_graph.add_vertices(g1.order())
    for v in new_graph.vertices():
        if vertex_assignments[v] == 0:
            for k in [k for k in g1.neighbors(v) if k > v]:
                new_graph.add_edge(v, k)
        else:
            for k in [k for k in g2.neighbors(v) if k > v]:
                new_graph.add_edge(v, k)

    new_graph, _ = remove_extra_edges(new_graph)
    assert (new_graph.order() == g1.order())
    return new_graph


@wrap_with_log
def cr6(g1, g2):
    """Orders the vertices of g1 and g2 according to their contribution to lovasz theta.
    Find subgraphs sg1 and sg2 such that sg1.order()+sg2.order()==g1.order()
                           and such that the total sum of contributions is maximized.
    add all edges between sg1 and sg2, then remove edges which don't affect the independence number.
    This function assumes g1 and g2 have the same number of vertices. Might fail otherwise.
    """
    assert (g1.order() == g2.order())
    costs_g1 = g1.vertex_cost_list()
    costs_g2 = g2.vertex_cost_list()
    index_g1 = 0
    index_g2 = 0
    while (index_g1 + index_g2 < g1.order()):  # after this loop,
        if costs_g1[index_g1][1] > costs_g2[index_g2][1]:
            index_g1 += 1
        else:
            index_g2 += 1
    sg1 = g1.subgraph([c[0] for c in costs_g1[:index_g1]])
    sg2 = g2.subgraph([c[0] for c in costs_g2[:index_g2]])

    # child_graph = sg1 + sg2  # These vertices are labeled [0..n]
    child_graph = sg1.disjoint_union(sg2)
    for v1, v2 in itertools.product(range(sg1.order()), range(sg2.order())):
        child_graph.add_edge(v1, v2 + sg1.order())
    child_graph, _ = remove_extra_edges(child_graph)
    assert child_graph.order() == g1.order()

    return child_graph


@wrap_with_log
def cr7(g1, g2):
    """Aligns the graphs according to vertex cost.
    When an edge is present in both graphs, we keep it.
    When it is only in one graph, we flip a coin.
    """
    assert (g1.order() == g2.order())
    costs_g1 = g1.vertex_cost_list()
    g1_new_order = [c[0] for c in costs_g1]  # list determines how to align the vertices of g1
    costs_g2 = g2.vertex_cost_list()
    g2_new_order = [c[0] for c in costs_g2]
    # g2_new_order.reverse()
    dict = {}
    for v in range(g1.order()):
        neighbors = []
        vertex_in_g1 = g1_new_order[v]
        vertex_in_g2 = g2_new_order[v]
        g1_neighbors = g1.neighbors(vertex_in_g1)
        g2_neighbors = g2.neighbors(vertex_in_g2)
        g1_neighbors = [a for a, b in enumerate(g1_new_order) if b in g1_neighbors]
        g2_neighbors = [a for a, b in enumerate(g2_new_order) if b in g2_neighbors]
        total_neighbors = g1_neighbors + g2_neighbors
        for t in total_neighbors:
            if t not in neighbors:
                if t in g1_neighbors and t in g2_neighbors:
                    neighbors.append(t)
                else:
                    r = np.random.rand()
                    if r > 0.5:
                        neighbors.append(t)
        dict[v] = neighbors
    new_graph = ExtendedGraph(edge_list_from_dict(dict))
    while (new_graph.order() < g1.order()):
        new_graph.add_vertex()
    assert (new_graph.order() == g1.order())
    return new_graph


@wrap_with_log
def cr8(g1, g2):
    """ adds an edge if there is an edge in g1 or in g2"""
    assert (g1.order() == g2.order())
    new_graph = g1.copy()
    for e in g2.edges():
        if not new_graph.has_edge(e[0], e[1]):
            new_graph.add_edges([e])
    new_graph, _ = remove_extra_edges(new_graph)
    assert (new_graph.order() == g1.order())
    return new_graph


@wrap_with_log
def cr_distinguished(g1, g2):
    """same as cr8 but for distinguished vertex"""
    assert (g1.order() == g2.order())
    new_graph = g1.copy()
    for e in g2.edges():
        if not new_graph.has_edge(e[0], e[1]):
            new_graph.add_edges([e])
    new_graph, _ = remove_extra_edges(new_graph, distinguished=True)
    assert (new_graph.order() == g1.order())
    return new_graph
# def cr_distinguished_vertex(g1,g2):
#     """Assumes that the subgraph induced by deleting the last vertex
#     (the distinguished one) from g1 and g2 is the same."""
#     g1_sub = g1.induced_subgraph(range(g1.order()), implementation = "copy_and_delete")
#     g2_sub = g2.induced_subgraph(range(g2.order()), implementation = "copy_and_delete")
#     assert g1_sub.EdgeSeq() == g2_sub.EdgeSeq() #checks that the edges are the same.
#     new_edges =
