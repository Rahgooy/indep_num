#!/usr/bin/python3
from algorithm_controller import incremental_ga
from caching import print_cache_stats
from logger import global_logger

if __name__ == "__main__":
    incremental_ga()
    global_logger.print_profile()
    print_cache_stats()
