import pyperf
import numpy as np
from septs import run_sept_suite
import subprocess


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def run_suite() -> None:
    inf = np.iinfo(np.int32)
    runner = pyperf.Runner()
    runner.metadata["git_commit_hash"] = get_git_revision_short_hash()
    run_sept_suite(runner=runner, rows=100000, cols=7, lower_bound=inf.min, upper_bound=inf.max)
    run_rept_suite(runner=runner, rept_dimension=1000, rows=100000, cols=7, lower_bound=inf.min, upper_bound=inf.max)

run_suite()