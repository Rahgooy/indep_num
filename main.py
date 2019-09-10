#!/usr/bin/python3
from algorithm_controller import search_with_vanguard, extend_search
from caching import print_cache_stats, reset_cache_number
from logger import global_logger
from functools import reduce
from start_graphs import *
import functions as FUN
import time
if __name__ == "__main__":
    # incremental_ga(initial_size = 12, final_size = 20,
    #                iterations = 10, iterations_between_updates = 2,
    #                pop_size = 100, independence_number=3)
    #result_table = open("multi_threaded_time_test.txt","w+")

    meta_pop = 4
    branch_factor = 5
    pop_per_mu = 150
    iterations_per_mu = 10
    elite_percent = 0.1
    crossover_percent = 0.2
    meta_elite_percent = 0.2
    make_unique = True
    #meta_select_procs=["make_extra_unique"]
    meta_select_proc = "take_best_very_unique"
    start_time = time.process_time()

    options = {"branch_factor":branch_factor, "meta_pop":meta_pop, "pop_per_mu":pop_per_mu, "iterations_per_mu":iterations_per_mu,
               "elite_percent":elite_percent, "crossover_percent":crossover_percent, "meta_elite_percent":meta_elite_percent,
               "make_unique":make_unique, "start_graph":start_graph_a3}
    #best_graph = search_with_vanguard(options)
    while(True):
        extend_search(options)
        # result_settings = reduce(lambda x,y: str(x)+ ","+str(y), [options[key] for key in options.keys()], "")
        # result_table.write(result_settings)
        # result_table.write(',')
        # result_table.write(str(FUN.fit(best_graph)) + "," + str(best_graph.lovasz_theta()) + "," + str(best_graph.alpha()))
        # result_table.write("\n")
    #print(best_graph.adjacency_matrix())
    #result_table.close()
    end_time = time.process_time()
    print("total time is ", end_time - start_time)
    # exit()
    # g = search_with_vanguard(options)
    # print (g.lovasz_theta())
    # print (g.alpha())
    # print (g.order())
    # print (g.adjacency_matrix())
    global_logger.print_profile()
    #print_cache_stats()
"""branch_factors = [1,2,5,7]
    meta_pops = [1,3,5,10,20]
    pop_per_mu = [50,100,200]
    iterations_per_mu = [5,10,15]
    elite_percent = [0.05]
    crossover_percent = [0,0.3]
    meta_elite_percent = [0.0, 0.2, 0.5]
    make_unique = [True, False]
    meta_select_proc = ["make_unique_then_select", "only_add_elites",
                        "make_extra_unique", "take_best", "take_best_unique"]
"""
