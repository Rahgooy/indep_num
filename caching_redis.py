from logger import global_logger as log
from logger import wrap_with_log
import redis
pool = redis.ConnectionPool(host='172.17.0.1')
red = redis.Redis(connection_pool = pool)

#CACHE = {"stats": {"largest_graph_size": 0}}
CACHE_ENABLED = True
CLEAR_SMALLER_GRAPHS = True
START_MATRICES={}
START_INDEP_SETS={}

def reset_redis_graph_statistics():
    """resets the amount of times each graph has been explored to 0."""

@wrap_with_log
def get_graphs_from_redis(graph_size, initial_graph):
    values = red.get(str(graph_size)+"-"+str(initial_graph.adjacency_matrix()))
    if values is None:
        return None
    else:
        return values #we will plan to call 'eval' on values when Extended Graph is available.

@wrap_with_log
def set_graphs_to_redis(tuples): #expect tuples to be [[graph, fitness, usage_number ]]
    graph_size = tuples[0][0].order()
    initial_graph = tuples[0][0].subgraph(range(10))
    value = "["
    for i in tuples:
        value += "[ExtendedGraph("+str(i[0].edges())+")," + str(i[1]) +"," + str(i[2]) +"],"
    value = value[:-1] +"]"
    red.set(str(graph_size)+"-"+str(initial_graph.adjacency_matrix()), value)

@wrap_with_log
def update_indep_set_batch(updates):
    pipe = red.pipeline()
    for u in updates:
        pipe.hset(u[0], u[1], str(u[2]))
    pipe.execute()

@wrap_with_log
def calculate_fitness_in_batch(pop):
    pipe = red.pipeline()
    for p in pop:
        pipe.hget(str(p.adjacency_matrix()), "lovasz_theta")

    responses = pipe.execute()

    response_pipe = red.pipeline()
    fitnesses = []

    newly_computed_graphs = {}
    for index in range(len(pop)):
        theta = responses[index]
        key = str(pop[index].adjacency_matrix())
        if theta is None:
            if key in newly_computed_graphs.keys():
                if 'theta' in newly_computed_graphs[key].keys():
                    theta = newly_computed_graphs[key]['theta']
                else:
                    theta = pop[index].raw_theta()
                    response_pipe.hset(key, "lovasz_theta", theta)
                    newly_computed_graphs[key].update({'theta': theta})
            else:
                theta = pop[index].raw_theta()
                response_pipe.hset(key, "lovasz_theta", theta)
                newly_computed_graphs[key] = {'theta': theta}
        else:
            theta = eval(theta)

        fitnesses.append(theta/4) #probably not good to hardcode this!
    # for f in fitnesses:
    #     assert not f is None
    response_pipe.execute()
    return fitnesses

    #calculates all of the fitnesses for a population of graphs in one query.

@wrap_with_log
def set_to_start_matrices(g):
    k = str(g.adjacency_matrix())
    result = g.calculate_start_matrix()
    START_MATRICES[k]=result


@wrap_with_log
def get_from_start_matrices(g):
    k = str(g.adjacency_matrix())
    if k in START_MATRICES.keys():
        return START_MATRICES[k]
    else:
        return None

@wrap_with_log
def set_to_start_indep_sets(g):
    k = str(g.adjacency_matrix())
    result = g.maximal_independent_vertex_sets()
    if len(result[0])==len(result[-1]):
        START_INDEP_SETS[k]=result

@wrap_with_log
def get_from_start_indep_sets(g):
    k = str(g.adjacency_matrix())
    if k in START_INDEP_SETS.keys():
        return START_INDEP_SETS[k]
    else:
        return None

#red.hset("stats", "largest_graph_size", 0)

def clear_cache():
    START_MATRICES={}
    START_INDEP_SETS={}
    #red.flushdb()

def wrap_extended_graph_method_with_cache(f):
    def wrap_cache(*args, **kwargs):
        g = args[0]
        k = str(g.adjacency_matrix())

        ans = red.hget(k,f.__name__)

        if not ans is None:
            #print("found old", f.__name__)
            return eval(ans)

        else:
            #print("set new", f.__name__)
            result = f(*args, **kwargs)
            red.hset(k, f.__name__, str(result))
            return result


    if CACHE_ENABLED:
        wrap_cache.__name__ = f.__name__
        return wrap_cache
    else:
        return f
