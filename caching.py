from logger import global_logger as log

CACHE = {"stats": {"largest_graph_size": 0}}
CACHE_ENABLED = True
CLEAR_SMALLER_GRAPHS = True
def get_cache_number():
    return CACHE["stats"]["largest_graph_size"]

def clear_cache():
    print("clearing")
    CACHE = {"stats": {"largest_graph_size": 0}}
    print(CACHE)

def check_start_matrix(graph):
    """returns the start matrix if available, else None"""
    k = str(graph.adjacency_matrix()).__hash__()
    if not k in CACHE.keys():
        print("missing key")
        return None
    elif not "calculate_start_matrix" in CACHE[k].keys():
        print("missing value")
        return None
    else:
        return CACHE[k]["calculate_start_matrix"]

def reset_cache_number():
    CACHE["stats"]["largest_graph_size"]=0

def clear_graphs(n):
    """Removes the graphs with n nodes or fewer."""
    to_remove = []
    for key in CACHE.keys():
        if key != "stats" and ( (not "lovasz_theta_and_cost_list" in CACHE[key]) or len(CACHE[key]["lovasz_theta_and_cost_list"][1])<=n):
            to_remove.append(key)
    for key in to_remove:
        del CACHE[key]

def print_cache_stats():
    log.writeln("=" * 85)
    log.writeln("=" + " " * 35 + " Cache Stats " + " " * 35 + "=")
    log.writeln("=" * 85)
    log.writeln(f'{"Function":40s} {"#calls":10s} {"Hit":10s} {"Miss":10s} {"Percentage":15s}')
    log.writeln("-" * 85)
    for key in CACHE["stats"]:
        if key != "largest_graph_size":
            hit = CACHE["stats"][key]["hit"]
            miss = CACHE["stats"][key]["miss"]
            p = hit / (hit + miss) * 100
            log.writeln(f'{key:40s} {hit+miss:<10d} {hit:<10d} {miss:<10d} {p:<15.2f}')
    log.writeln("=" * 85)


def wrap_extended_graph_method_with_cache(f):
    def wrap_cache(*args, **kwargs):
        g = args[0]
        k = str(g.adjacency_matrix()).__hash__()
        if k not in CACHE:
            CACHE[k] = {}
        if f.__name__ not in CACHE["stats"]:
            CACHE["stats"][f.__name__] = {"hit": 0, "miss": 0}
        if f.__name__ not in CACHE[k]:
            CACHE[k][f.__name__] = f(*args, **kwargs)
            CACHE["stats"][f.__name__]["miss"] += 1
            if CLEAR_SMALLER_GRAPHS and f.__name__== "lovasz_theta_and_cost_list" and len(CACHE[k][f.__name__][1]) > CACHE["stats"]["largest_graph_size"]:
                print (len(CACHE.keys()))
                print ("clearing")
                print (CACHE["stats"]["largest_graph_size"])
                to_return = CACHE[k][f.__name__]
                CACHE["stats"]["largest_graph_size"]=len(CACHE[k][f.__name__][1])
                clear_graphs(len(CACHE[k][f.__name__][1])-2)
                print (len(CACHE.keys()))
                print (CACHE["stats"]["largest_graph_size"])
                return to_return
        else:
            CACHE["stats"][f.__name__]["hit"] += 1

        return CACHE[k][f.__name__]

    if CACHE_ENABLED:
        wrap_cache.__name__ = f.__name__
        return wrap_cache
    else:
        return f
