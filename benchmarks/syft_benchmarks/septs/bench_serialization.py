import syft as sy
from .util import make_sept, generate_data
import pyperf
import functools

def create_bench_sept_serialize(runner: pyperf.Runner, rows: int, columns: int, lower_bound: int, upper_bound: int):
    data = generate_data(rows, columns, lower_bound, upper_bound)
    sept = make_sept(data, lower_bound, upper_bound)
    partially_evaluated_func = functools.partial(sy.serialize, sept, True)
    runner.bench_func(f"serialize_sept_rows_{rows}_columns_{columns}", partially_evaluated_func)