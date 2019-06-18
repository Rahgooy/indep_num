import sys
import time


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
                'total': 0
            }
        self.profile[fname]['current'] = time.clock()

    def finish(self, fname):
        if self.log_profiling:
            self.info(f"{fname} is Finished.")
        self.profile[fname]['total'] += time.clock() - self.profile[fname]['current']

    def print_profile(self):
        for k in self.profile:
            self.out.write(f"{k} : {self.profile[k]['total']:0.2f} seconds\n")

    def __call__(self, *args, **kwargs):
        self.info(*args)

    def info(self, msg):
        self.out.write('\033[32m' + time.asctime().split()[3] + ' - ' + '\033[35m' + msg + '\033[0m' + '\n')


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
