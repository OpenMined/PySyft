# third party
import pyperf

# relative
from ..common_constructor import create_bench_constructor
from .bench_deserialization import create_bench_phitensor_deserialize
from .bench_serialization import create_bench_phitensor_serialize
from .util import make_phitensor


def run_phitensor_suite(
    runner: pyperf.Runner,
    data_file,
) -> None:
    phitensor = make_phitensor(data_file)
    create_bench_constructor(runner, data_file=data_file)
    create_bench_phitensor_serialize(runner, phitensor)
    create_bench_phitensor_deserialize(runner, phitensor)
