from logger import global_logger as log

import redis
import ast
red = redis.Redis(host="172.17.0.1")

#CACHE = {"stats": {"largest_graph_size": 0}}
START_MATRICES={}

@wrap_with_log
def calculate_fitness_in_batch(pop):
    pipe = red.pipeline()
    for p in pop:
        pipe.hget(str(p.adjacency_matrix()), "lovasz_theta")
    for p in pop:
        pipe.hget(str(p.adjacency_matrix()), "maximal_independent_vertex_sets")
    responses = pipe.execute()

    response_pipe = red.pipeline()
    fitnesses = []
    for index in range(len(pop)):
        theta = responses[index]
        if theta is None:
            theta = pop[index].raw_theta()
            response_pipe.hset(str(pop[index]), "lovasz_theta", theta)
        if responses[index+len(pop)] is None:
            indep_sets = pop[index].raw_maximal_independent_vertex_sets()
            response_pipe.hset(str(pop[index]), "maximal_independent_vertex_sets", str(indep_sets))
            independence_number = len(indep_sets[-1])
        else:
            independence_number = len(eval(responses[index+len(pop)])[-1])
        fitnesses.append(theta/independence_number)
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

red.hset("stats", "largest_graph_size", 0)
CACHE_ENABLED = True
CLEAR_SMALLER_GRAPHS = True

def clear_cache():
    START_MATRICES={}
    red.flushdb()


def wrap_extended_graph_method_with_cache(f):
    def wrap_cache(*args, **kwargs):
        g = args[0]
        k = str(g.adjacency_matrix())

        ans = red.hget(k,f.__name__)

        if not ans is None:
            #print("found old")
            return eval(ans)

        else:
            #print("set new")
            result = f(*args, **kwargs)
            red.hset(k, f.__name__, str(result))
            return result


    if CACHE_ENABLED:
        wrap_cache.__name__ = f.__name__
        return wrap_cache
    else:
        return f
