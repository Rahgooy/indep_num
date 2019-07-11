#!/usr/bin/python3
from algorithm_controller import incremental_ga
from algorithm_controller import search_with_vanguard
from caching import print_cache_stats
from logger import global_logger

if __name__ == "__main__":
    # incremental_ga(initial_size = 12, final_size = 20,
    #                iterations = 10, iterations_between_updates = 2,
    #                pop_size = 100, independence_number=3)
    g = search_with_vanguard()
    print (g.lovasz_theta())
    print (g.alpha())
    print (g.order())
    global_logger.print_profile()
    print_cache_stats()
