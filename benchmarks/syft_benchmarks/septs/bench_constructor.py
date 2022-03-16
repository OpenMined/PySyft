# stdlib
import functools

# third party
import numpy as np
import pyperf

# relative
from .util import make_sept


def create_bench_sept_constructor(
    runner: pyperf.Runner, rows: int, cols: int, lower_bound: int, upper_bound: int
) -> None:
    data = np.random.randint(
        lower_bound, upper_bound, size=(rows, cols), dtype=np.int32
    )
    partially_evaluated_func = functools.partial(
        make_sept, data, lower_bound, upper_bound
    )
    runner.bench_func(
        f"constructor_sept_rows_{rows}_cols_{cols}", partially_evaluated_func
    )
