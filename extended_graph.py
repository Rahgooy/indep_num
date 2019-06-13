from igraph import *


class ExtendedGraph(Graph):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def lovasz_theta(self):
        # TODO: implement this function
        return 1
