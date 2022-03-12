# third party
import pyperf

# relative
from ..common_constructor import create_bench_constructor
from .bench_deserialization import create_bench_rept_deserialize
from .bench_serialization import create_bench_rept_serialize
from .util import make_ndept


def run_ndept_suite(
    runner: pyperf.Runner,
    data_file,
) -> None:
    ndept = make_ndept(data_file)
    create_bench_constructor(runner, data_file=data_file, ndept=True)
    create_bench_rept_serialize(runner, ndept)
    create_bench_rept_deserialize(runner, ndept)
