# stdlib
import functools

# third party
import pyperf

# syft absolute
import syft as sy

# relative
from .util import generate_data
from .util import make_sept


def create_bench_sept_deserialize(
    runner: pyperf.Runner, rows: int, columns: int, lower_bound: int, upper_bound: int
) -> None:
    data = generate_data(rows, columns, lower_bound, upper_bound)
    serialized_sept = sy.serialize(
        make_sept(data, lower_bound, upper_bound), to_bytes=True
    )
    partialy_evaluated_func = functools.partial(
        sy.deserialize, serialized_sept, from_bytes=True
    )
    runner.bench_func(
        f"deserialize_sept_rows_{rows}_columns_{columns}", partialy_evaluated_func
    )
