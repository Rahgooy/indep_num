from igraph import *

from logger import wrap_with_log
from lovasz import lovasz_theta
from caching_redis import wrap_extended_graph_method_with_cache as wrap_with_cache
from caching_redis import get_from_start_matrices, get_from_start_indep_sets
from numpy.random import randint, rand
import cvxopt
import functions as FUN
import numpy as np
class ExtendedGraph(Graph):
    def __init__(self, *args, **kwds):
        super(ExtendedGraph, self).__init__(args[0])



    @wrap_with_log
    def calculate_start_matrix(self):
        print("changes made")
        B = lovasz_theta(self, long_return = True)['B']
        BB = [row.tolist() + [0] for row in B]
        BB.append([0]*len(BB)+[1])
        result = 0.75*np.array(BB)+0.25*np.identity(len(BB))
        return result

    @wrap_with_log
    @wrap_with_cache
    def lovasz_theta(self):
        subgraph = self.induced_subgraph(range(self.order()-1), implementation="copy_and_delete")
        seed = get_from_start_matrices(subgraph)
        if not seed is None:
            return lovasz_theta(self, start = {'zs':[cvxopt.matrix(seed)]})
        else:
            return lovasz_theta(self)
        #return self.lovasz_theta_and_cost_list()[0]

    @wrap_with_log
    def raw_theta(self):
        subgraph = self.induced_subgraph(range(self.order()-1), implementation="copy_and_delete")
        seed = get_from_start_matrices(subgraph)
        seed=None
        if not seed is None:
            return lovasz_theta(self, start = {'zs':[cvxopt.matrix(seed)]})
        else:
            return lovasz_theta(self)

    @wrap_with_log
    def maximal_independent_vertex_sets(self):
        subgraph = self.induced_subgraph(range(self.order()-1), implementation= "copy_and_delete")
        seed = get_from_start_indep_sets(subgraph)
        if not seed is None:
            return FUN.calculate_independent_sets_from_subgraph(seed,self)
        else:
            return self.raw_maximal_independent_vertex_sets()
        # independent_sets = sorted(super().maximal_independent_vertex_sets(), key=len)
        # independent_sets = [set(i) for i in independent_sets]
        # return independent_sets# indep_num, independent_sets

    @wrap_with_log
    def raw_maximal_independent_vertex_sets(self):
        #"""computes the pair independence_number, maximal independent sets"""
        independent_sets = sorted(super().maximal_independent_vertex_sets(), key=len)
        independent_sets = [set(i) for i in independent_sets]
        return independent_sets# indep_num, independent_sets

    @wrap_with_log
    def independence_number(self):
        return super().independence_number()
        #return len(self.maximal_independent_vertex_sets()[-1])
        #return self.independence_number_and_maximal_independent_sets()[0]

    @wrap_with_log
    def largest_independent_vertex_sets(self):
        alpha = self.independence_number()
        return [s for s in self.maximal_independent_vertex_sets() if len(s)==alpha ]


    def vertices(self):
        return self.vs.indices

    def edges(self):
        return [(e.source, e.target) for e in EdgeSeq(self)]

    def order(self):
        return self.vcount()

    def adjacency_matrix(self):
        return self.get_adjacency().data

    def has_edge(self, u, v):
        if u < v:
            return (u, v) in self.get_edgelist()
        else:
            return (v, u) in self.get_edgelist()

    def size(self):
        return len(self.get_edgelist())

    def asExtended(g):
        new_g = ExtendedGraph(g.vcount())
        new_g.add_edges(g.get_edgelist())
        return new_g

    def lift(self,flipped_edges):
        size = self.vcount()
        lifted_graph = self.disjoint_union(self)
        edges_to_delete = flipped_edges + [(e1+size, e2+size) for e1,e2 in flipped_edges]
        edges_to_add = [(e1, e2+size) for e1, e2 in flipped_edges] + [(e1+size, e2) for e1, e2 in flipped_edges]
        lifted_graph.delete_edges(edges_to_delete)
        lifted_graph.add_edges(edges_to_add)
        return lifted_graph

    def random_lift(self):
        flipped_edges = [e for e in self.edges() if np.random.rand() >0.5]
        return self.lift(flipped_edges)

    def co_lift(self):
        lift = self.complementer()
        lift.simplify()
        lift = lift.random_lift()
        lift = lift.complementer()
        lift.simplify()
        lift, _ = FUN.remove_extra_edges(lift)
        return lift
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
        g = ExtendedGraph(FUN.edge_list_from_dict(dict))
        while (g.order() < n):  # this can occur if the random graph is not connected
            g.add_vertex()
        assert g.order() == n
        return g

        """---"""

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
        r_graph = ExtendedGraph(FUN.edge_list_from_dict(g))
        while r_graph.order() < n:
            r_graph.add_vertex()
        return r_graph
