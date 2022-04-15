# stdlib
import subprocess

# third party
import numpy as np
import pyperf
from syft_benchmarks import run_rept_suite
from syft_benchmarks import run_sept_suite


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def run_suite() -> None:
    inf = np.iinfo(np.int32)
    runner = pyperf.Runner()
    runner.parse_args()
    runner.metadata["git_commit_hash"] = get_git_revision_short_hash()
    run_sept_suite(
        runner=runner, rows=1000, cols=10, lower_bound=inf.min, upper_bound=inf.max
    )
    run_rept_suite(
        runner=runner,
        rept_dimension=15,
        rows=1000,
        cols=10,
        lower_bound=inf.min,
        upper_bound=inf.max,
    )


run_suite()
