from igraph import *
from lovasz import lovasz_theta


class ExtendedGraph(Graph):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def lovasz_theta(self):
        return lovasz_theta(self)

    def vertices(self):
        return self.vs.indices

    def adjacency_matrix(self):
        return self.get_adjacency().data

    def has_edge(self, u, v):
        return (u, v) in self.get_adjedgelist()

    def asExtended(g):
        new_g = ExtendedGraph(g.vcount())
        new_g.add_edges(g.get_edgelist())
        return new_g
