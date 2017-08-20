from time import time
from line_profiler import LineProfiler


class Benchmark():
    """
    This is a testing class for the benchmarking of functions.
    Input of the Benchmark class should be the function to test
    and the params (optional)
    """

    def __init__(self, function, **params):
        self.function = function
        self.params = params

    def exec_time(self, reps=1):
        """
        Calls function x-times and returns an array of computed execution times
        """
        results = []
        for rep in range(reps):
            t0 = time()
            self.function(**self.params)
            t1 = time()
            results.append(t1 - t0)
        return results

    def profile_lines(self):
        """
        A simple wrapper to call the line_profiler.
        Prints the line_profiler output
        """
        lp = LineProfiler()
        lp_wrapper = lp(self.function)
        lp_wrapper(**self.params)
        lp.print_stats()
