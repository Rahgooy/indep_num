from extended_graph import *
"""this file defines the start graphs for the algorithm."""
start_graph_a2 = ExtendedGraph([(i,(i+1)%5) for i in range(5)]) #the 5 cycle
start_graph_a3 = ExtendedGraph([(i,i+1) for i in range(4)] + [(4,0),(5,4),(5,2)]) #alpha = 3 Original rdb.
start_graph_a4 = start_graph_a2.disjoint_union(start_graph_a2)
