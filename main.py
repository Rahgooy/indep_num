#!/usr/bin/python3
from algorithm_controller import search_with_vanguard
from caching import print_cache_stats, reset_cache_number
from logger import global_logger
from itertools import product
from caching import clear_cache
from functools import reduce
import functions as FUN
if __name__ == "__main__":
    # incremental_ga(initial_size = 12, final_size = 20,
    #                iterations = 10, iterations_between_updates = 2,
    #                pop_size = 100, independence_number=3)
    result_table = open("parameter_tuning_pt11_dontremovextraedges.txt","w+")

    meta_pops = [10]
    branch_factors = [3,5,7]
    pop_per_mus = [50,100]
    iterations_per_mus = [10]
    elite_percents = [0.5, 0.3, 0.1]
    crossover_percents = [0.5, 0.3, 0.1]
    meta_elite_percents = [0.2]
    make_uniques = [True]
    #meta_select_procs=["make_extra_unique"]
    meta_select_procs = ["take_best_very_unique"]

    for branch_factor, meta_pop, pop_per_mu, iterations_per_mu, elite_percent, crossover_percent, meta_elite_percent, make_unique, meta_select_proc in product(branch_factors, meta_pops, pop_per_mus, iterations_per_mus,
                                                                            elite_percents, crossover_percents, meta_elite_percents, make_uniques, meta_select_procs):
        options = {"branch_factor":branch_factor, "meta_pop":meta_pop, "pop_per_mu":pop_per_mu, "iterations_per_mu":iterations_per_mu,
                   "elite_percent":elite_percent, "crossover_percent":crossover_percent, "meta_elite_percent":meta_elite_percent,
                   "make_unique":make_unique, "meta_select_proc":meta_select_proc}
        if not(options["meta_elite_percent"]* options["meta_pop"]<1.0 and options["meta_select_proc"] == "only_add_elites"):
            best_graph = search_with_vanguard(options)
            result_settings = reduce(lambda x,y: str(x)+ ","+str(y), [options[key] for key in options.keys()], "")
            result_table.write(result_settings)
            result_table.write(',')
            result_table.write(str(FUN.fit(best_graph)) + "," + str(best_graph.lovasz_theta()) + "," + str(best_graph.alpha()))
            result_table.write("\n")
            #print (branch_fator, meta_pop, pop_per_mu, iterations_per_mu, elite_percent, crossover_percent, meta_elite_percent, make_unique, meta_select_proc)
            global_logger.reset() #reset logger
            reset_cache_number()
    result_table.close()
    # exit()
    # g = search_with_vanguard(options)
    # print (g.lovasz_theta())
    # print (g.alpha())
    # print (g.order())
    # print (g.adjacency_matrix())
    # global_logger.print_profile()
    print_cache_stats()
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
