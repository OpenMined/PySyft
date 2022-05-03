# stdlib
import functools

# third party
import pyperf

# syft absolute
import syft as sy


def create_bench_ndept_serialize(runner: pyperf.Runner, ndept):
    partially_evaluated_func = functools.partial(sy.serialize, ndept, to_bytes=True)
    runner.bench_func(
        f"serialize_ndept__{len(ndept):,}",
        partially_evaluated_func,
    )
