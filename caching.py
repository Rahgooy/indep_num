from logger import global_logger as log

CACHE = {"stats": {}}
CACHE_ENABLED = True


def print_cache_stats():
    log.writeln("=" * 85)
    log.writeln("=" + " " * 35 + " Cache Stats " + " " * 35 + "=")
    log.writeln("=" * 85)
    log.writeln(f'{"Function":40s} {"#calls":10s} {"Hit":10s} {"Miss":10s} {"Percentage":15s}')
    log.writeln("-" * 85)
    for key in CACHE["stats"]:
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
        else:
            CACHE["stats"][f.__name__]["hit"] += 1

        return CACHE[k][f.__name__]

    if CACHE_ENABLED:
        wrap_cache.__name__ = f.__name__
        return wrap_cache
    else:
        return f
