# stdlib
import functools

# third party
import pyperf

# syft absolute
import syft as sy


def create_bench_phitensor_serialize(runner: pyperf.Runner, phitensor):
    partially_evaluated_func = functools.partial(sy.serialize, phitensor, to_bytes=True)
    runner.bench_func(
        f"serialize_phitensor__{len(phitensor):,}",
        partially_evaluated_func,
    )
