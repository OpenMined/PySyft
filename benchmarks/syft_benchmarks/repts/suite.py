# third party
import pyperf
import numpy as np

# relative
from .bench_constructor import create_bench_constructor
from .bench_deserialization import create_bench_rept_deserialize
from .bench_serialization import create_bench_rept_serialize


def run_rept_suite(
    runner: pyperf.Runner,
    rept_dimension: int = 15,
    rows: int = 10,
    cols: int = 100,
    lower_bound: int = np.iinfo(np.int32).min,
    upper_bound: int = np.iinfo(np.int32).max,
) -> None:
    create_bench_constructor(
        runner, rept_dimension, rows, cols, lower_bound, upper_bound
    )
    create_bench_rept_serialize(
        runner, rept_dimension, rows, cols, lower_bound, upper_bound
    )
    create_bench_rept_deserialize(
        runner, rept_dimension, rows, cols, lower_bound, upper_bound
    )
