# stdlib
import functools

# third party
import pyperf

# syft absolute
import syft as sy


def create_bench_phitensor_deserialize(runner: pyperf.Runner, phitensor):
    serialized_data = sy.serialize(phitensor, to_bytes=True)
    partially_evaluated_func = functools.partial(
        sy.deserialize, serialized_data, from_bytes=True
    )
    runner.bench_func(
        f"deserialize_phitensor__{len(phitensor):,}",
        partially_evaluated_func,
    )
