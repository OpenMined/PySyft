# third party
import pyperf

# relative
from .bench_constructor import create_bench_sept_constructor
from .bench_deserialization import create_bench_sept_deserialize
from .bench_serialization import create_bench_sept_serialize


def run_sept_suite(runner: pyperf.Runner, rows, cols, lower_bound, upper_bound):
    create_bench_sept_deserialize(runner, rows, cols, lower_bound, upper_bound)
    create_bench_sept_constructor(runner, rows, cols, lower_bound, upper_bound)
    create_bench_sept_serialize(runner, rows, cols, lower_bound, upper_bound)
