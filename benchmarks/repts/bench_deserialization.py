import syft as sy
from .util import make_rept
import pyperf
import functools


def create_bench_rept_deserialize(runner: pyperf.Runner, rept_dimension: int, rows: int, columns: int, lower_bound: int, upper_bound: int):
    rept = make_rept(rept_dimension, rows, columns, lower_bound, upper_bound)
    serialized_data = sy.serialize(rept, to_bytes=True)
    partially_evaluated_func = functools.partial(sy.deserialize, serialized_data, True)
    runner.bench_func(f"deserialize_rept__rept_dimension_{rept_dimension}_rows_{rows}_columns_{columns}", partially_evaluated_func)