from igraph import *

from logger import wrap_with_log
from lovasz import lovasz_theta
from caching import wrap_extended_graph_method_with_cache as wrap_with_cache


class ExtendedGraph(Graph):
    def __init__(self, *args, **kwds):
        super(ExtendedGraph, self).__init__(args[0])
        self.cache = {}
        self.key = lambda g: str(g.adjacency_matrix()).__hash__()

    @wrap_with_log
    @wrap_with_cache
    def lovasz_theta(self):
        return lovasz_theta(self)

    @wrap_with_log
    @wrap_with_cache
    def independence_number(self):
        return super().independence_number()

    @wrap_with_log
    @wrap_with_cache
    def largest_independent_vertex_sets(self):
        return super().largest_independent_vertex_sets()

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
