from igraph import *

from logger import wrap_with_log
from lovasz import lovasz_theta
from caching import wrap_extended_graph_method_with_cache as wrap_with_cache
from caching import check_start_matrix, get_cache_number
import cvxopt
import functions as FUN
import numpy as np
class ExtendedGraph(Graph):
    def __init__(self, *args, **kwds):
        super(ExtendedGraph, self).__init__(args[0])

    @wrap_with_log
    @wrap_with_cache
    def calculate_start_matrix(self):
        print("changes made")
        B = lovasz_theta(self, long_return = True)['B']
        BB = [row.tolist() + [0] for row in B]
        BB.append([0]*len(BB)+[1])
        return 0.75*np.array(BB)+0.25*np.identity(len(BB))

    @wrap_with_log
    @wrap_with_cache
    def lovasz_theta_and_cost_list(self):
        subgraph = self.subgraph(range(self.order()-1), implementation = "copy_and_delete")
        assert subgraph.order() == self.order() -1
        seed = subgraph.calculate_start_matrix()
        """returns a pair, theta and the cost list."""
        order = self.order()
        assert order >= get_cache_number()
        #solution = LOV.lovasz_theta(self, long_return=True)
        if seed is None:
            print("no seed")
            print(self.order())
            print(self.adjacency_matrix())
            solution = lovasz_theta(self, long_return = True)
        else:
            #print("seeding")
            solution = lovasz_theta(self, long_return = True, start = {'zs':[cvxopt.matrix(seed)]})
        theta = solution['theta']
        witness = solution['B']
        costs = np.diagonal(witness) * theta
        costs = enumerate(costs)  # adds an index
        costs = sorted(costs, key=lambda x: -x[1])  # sort by the cost
        assert (order == self.order())
        return theta, costs
    @wrap_with_log
    def lovasz_theta(self):
        return self.lovasz_theta_and_cost_list()[0]
    @wrap_with_log
    def vertex_cost_list(self):
        return self.lovasz_theta_and_cost_list()[1]

    @wrap_with_log
    @wrap_with_cache
    def independence_number_and_maximal_independent_sets(self):
        """computes the pair independence_number, maximal independent sets"""
        independent_sets = super().maximal_independent_vertex_sets()
        independent_sets.sort(key = lambda x: -len(x))
        indep_num = len(independent_sets[0])
        return indep_num, independent_sets

    @wrap_with_log
    def independence_number(self):
        return self.independence_number_and_maximal_independent_sets()[0]

    @wrap_with_log
    def largest_independent_vertex_sets(self):
        alpha = self.independence_number()
        return [s for s in self.independence_number_and_maximal_independent_sets()[1] if len(s)==alpha ]
    @wrap_with_log
    def maximal_independent_vertex_sets(self):
        return self.independence_number_and_maximal_independent_sets()[1]

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

    """---"""
