import sys
import time
from helpers import OutputColors as clr


class Logger:

    def __init__(self, out_stream, log_profiling=False):
        self.out = out_stream
        self.log_profiling = log_profiling
        self.profile = {}

    def start(self, fname):
        if self.log_profiling:
            self.info(f"{fname} is statrted")

        if fname not in self.profile:
            self.profile[fname] = {
                'current': None,
                'total': 0,
                'calls': 0
            }
        self.profile[fname]['current'] = time.clock()
        self.profile[fname]['calls'] += 1

    def finish(self, fname):
        if self.log_profiling:
            self.info(f"{fname} is Finished.")
        self.profile[fname]['total'] += time.clock() - self.profile[fname]['current']

    def print_profile(self):
        self.out.write(clr.BLUE)
        self.out.write("=" * 85 + "\n")
        self.out.write("=" + " " * 35 + "Time Profiles" + " " * 35 + "=\n")
        self.out.write("=" * 85 + "\n")
        self.out.write(f"{'Function name':40s} {'# of calls':15s} {'time per call':15s} {'total time':15s}\n")
        for k in self.profile:
            self.out.write(f"{k:40s} {self.profile[k]['calls']:<15d} ")
            self.out.write(f"{self.profile[k]['total']/self.profile[k]['calls']:<15.2f} ")
            self.out.write(f"{self.profile[k]['total']:<15.2f}\n")
        self.out.write("=" * 85 + "\n")
        self.out.write(clr.RESET)

    def __call__(self, *args, **kwargs):
        self.info(*args)

    def info(self, msg):
        self.out.write(clr.GREEN)
        self.out.write(time.asctime().split()[3] + ' - ')
        self.out.write(clr.MAGENTA)
        self.out.write(clr.MAGENTA + msg + '\n')
        self.out.write(clr.RESET)


def wrap_with_log(logger, obj, names):
    """Wraps specified functions of an object with logger.start and logger.finish"""

    def wrap(func):
        def call(*args, **kwargs):
            logger.start(func.__name__)
            result = func(*args, **kwargs)
            logger.finish(func.__name__)
            return result

        return call

    for f in names:
        wrapped = wrap(getattr(obj, f))
        setattr(obj, f, wrapped)
