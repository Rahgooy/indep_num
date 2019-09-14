from extended_graph import *
"""this file defines the start graphs for the algorithm."""
trivial_graph = ExtendedGraph([])
trivial_graph.add_vertex()
#start_graph_a2 = trivial_graph.disjoint_union(trivial_graph)
start_graph_a2 = ExtendedGraph([(i,(i+1) %5) for i in range(5)] + [(0,2)])
start_graph_a3 = ExtendedGraph([(i,i+1) for i in range(4)] + [(4,0),(5,4),(5,2)]) #alpha = 3 Original rdb.
start_graph_a4 = ExtendedGraph([(i,(i+1) %5) for i in range(5)] + [(5+i,5+(i+1)%5) for i in range(5)] +[(i,i+5)for i in range(5)])
