# stdlib
import functools

# third party
import pyperf

# syft absolute
import syft as sy


def create_bench_rept_deserialize(runner: pyperf.Runner, ndept):

    serialized_data = sy.serialize(ndept, to_bytes=True)
    partially_evaluated_func = functools.partial(
        sy.deserialize, serialized_data, from_bytes=True
    )
    runner.bench_func(
        f"deserialize_ndept__{len(ndept):,}",
        partially_evaluated_func,
    )
