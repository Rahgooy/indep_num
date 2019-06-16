from igraph import *
from lovasz import lovasz_theta


class ExtendedGraph(Graph):
    def __init__(self, *args, **kwds):
        #print args[0]
        #g = Graph(args[0])
        #print g
        super(ExtendedGraph, self).__init__(args[0])

    def lovasz_theta(self):
        return lovasz_theta(self)

    def vertices(self):
        return self.vs.indices

    def edges(self):
        #print self.es
        #print [(e.source, e.target) for e in EdgeSeq(self)]
        return [(e.source, e.target) for e in EdgeSeq(self)]
    def order(self):
        return self.vcount()

    def adjacency_matrix(self):
        return self.get_adjacency().data

    def has_edge(self, u, v):
        return (u, v) in self.get_adjedgelist()

    def asExtended(g):
        new_g = ExtendedGraph(g.vcount())
        new_g.add_edges(g.get_edgelist())
        return new_g
