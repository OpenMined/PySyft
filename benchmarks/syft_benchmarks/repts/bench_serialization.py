import syft as sy
from .util import make_rept
import pyperf
import functools


def create_bench_rept_serialize(runner: pyperf.Runner, rept_dimension: int, rows: int, columns: int, lower_bound: int, upper_bound: int):
    rept = make_rept(rept_dimension, rows, columns, lower_bound, upper_bound)
    partially_evaluated_func = functools.partial(sy.serialize, rept, True)
    runner.bench_func(f"serialize_rept__rept_dimension_{rept_dimension}_rows_{rows}_columns_{columns}", partially_evaluated_func)