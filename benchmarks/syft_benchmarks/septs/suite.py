# third party
import pyperf
import numpy as np

# relative
from .bench_constructor import create_bench_sept_constructor
from .bench_deserialization import create_bench_sept_deserialize
from .bench_serialization import create_bench_sept_serialize


def run_sept_suite(runner: pyperf.Runner, rows: int = 10, cols: int = 1000, lower_bound: int = np.iinfo(np.int32).min, upper_bound: int = np.iinfo(np.int32).max):
    create_bench_sept_deserialize(runner, rows, cols, lower_bound, upper_bound)
    create_bench_sept_constructor(runner, rows, cols, lower_bound, upper_bound)
    create_bench_sept_serialize(runner, rows, cols, lower_bound, upper_bound)
