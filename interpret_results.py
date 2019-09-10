from caching_redis import get_graphs_from_redis
from extended_graph import *
from lovasz import lovasz_theta
import functools
import matplotlib
import matplotlib.pyplot as plt

base_graph = ExtendedGraph([(i,i+1) for i in range(4)] + [(4,0),(5,4),(5,2)])

def get_values_at_level(i):
    redis_values = get_graphs_from_redis(i,base_graph)
    if not redis_values is None:
        redis_values = eval(redis_values)
        redis_values.sort(key=lambda x: x[1], reverse = True)
    return redis_values

def create_plot():
    to_plotx = []
    to_ploty = []
    i = 6
    graphs_at_level = get_values_at_level(i)
    while(graphs_at_level is not None):
        # if check_state_independence(graphs_at_level[0][0]):
        #     print("state dependent contextuality at level ", i)ww

        to_plotx.append(i)
        to_ploty.append(graphs_at_level[0][1])
        i +=1
        graphs_at_level = get_values_at_level(i)

    fig, ax = plt.subplots()
    ax.plot(to_plotx, to_ploty)
    plt.xlabel("number of vertices")
    plt.ylabel("theta/alpha")
    plt.title("best graphs from start_graph_a2")
    fig.savefig("9_4_2019.png")

def necessary_check_for_sic(graph):
    a = graph.independence_number()
    w = graph.clique_number()
    n = graph.order()
    if n/(w+1) > a:
        print (n/(w+1), a)
        return True
    else:
        return False

def _check_state_independence(vectors, indep_num):
    vectors = map(lambda x: np.array(x)/np.linalg.norm(x), vectors)
    #outer_matrix = functools.reduce(lambda x,y: x+y, [np.outer(v1,v1) for v1 in vectors], np.zeros(np.shape(vectors[0])[0],np.shape(vectors[0])[0]) )
    outer_matrix = np.zeros((30,30))
    for v in vectors:
        outer_matrix += np.outer(v,v)
    vectors = np.array(vectors)
    print (np.linalg.eigh(outer_matrix)[0][0], indep_num)
    return np.linalg.eigh(outer_matrix)[0][0] > indep_num
def check_state_independence(graph):
    indep_num = graph.independence_number()
    B = lovasz_theta(graph, long_return = True)['B']
    C = np.linalg.cholesky(B)
    vectors = [v for v in C]
    return(_check_state_independence(vectors, indep_num))
def test_example():
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()

    fig.savefig("tet.png")
    plt.show()
