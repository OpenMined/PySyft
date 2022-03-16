# stdlib
import os
from pathlib import Path
import subprocess
from typing import Any
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple
from typing import cast
import sys
import os
import argparse

# third party
import pyperf
from syft_benchmarks import run_rept_suite
from syft_benchmarks import run_sept_suite
import click
import fire


@click.group()
def cli() -> None:
    pass

def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

# check what is action=store_true in pyperf

# @click.command()
# # @click.option('--rigorous', help="Spend longer running tests to get more accurate results")
# @click.option('--fast', is_flag=True, help="Get rough answers quickly")
# # @click.option('--debug-single-value', help="Debug mode, only compute a single value")
# # @click.option('-p', '--processes', help='number of processes used to run benchmarks ') # TODO add default processes
# @click.option('-n', '--values', type=int,help='number of values per process') # TODO add default values
# # @click.option('-w', '--warmups', help='number of skipped values per run used to warmup the benchmark')
# # @click.option('-l', '--loops', help='number of loops per value, 0 means automatic calibration ') # TODO add default
# # @click.option('-v', '--verbose', help='enable verbose mode')
# # @click.option('-q', '--quiet', help='enable quiet mode')
# # @click.option('--pipe', help='Write benchmarks encoded as JSON into the pipe FD')
# @click.option('-o', '--output', help='write results encoded to JSON into FILENAME')
# # @click.option('--append', help='append results encoded to JSON into FILENAME')
# # @click.option('--min-time', help='Minimum duration in seconds of a single value, used to calibrate the number of loops (default') # TODO add default
# # @click.option('--worker')
def run_suite() -> None:
    # print(sys.argv)
    # print(kwargs)
    inf = np.iinfo(np.int32)
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--test')
    runner = pyperf.Runner()
    print(sys.argv)
    runner.parse_args()
    runner.metadata["git_commit_hash"] = get_git_revision_short_hash()

    run_phitensor_suite(runner=runner, data_file=data_file)

# cli.add_command(run_suite)

if __name__ == "__main__":
    # cli()
    # fire.Fire(run_suite)
    run_suite()

