"""
This file can be used to parametrize the benchmarking of autodp tensor operations
This file can be run as:
python executor.py MODE_OF_OPERATION SUITE_ARGS PERF_ARGS

Parameters
----------
MODE_OF_OPERATION: {rept, sept, all}
    This option allow you to choose whether you want to run the rept tests or the septs test or both
SUITE_ARGS:
    These arguments control the size of the generated data used in our tests
PERF_ARGS:
    These arguments are inherited from the pyperf runner class,
    for more info: https://pyperf.readthedocs.io/en/latest/runner.html

"""
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
import numpy as np

# third party
import pyperf
from syft_benchmarks import run_rept_suite
from syft_benchmarks import run_sept_suite


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument(
        "--run",
        choices=["rept", "sept", "all"],
        help="Choice whether you want to test the Single Entity Phi Tensor or the Row Entity Phi Tensor or both",
    )
    subparsers = parser.add_subparsers(dest="run_type")
    subparsers.required = True

    parser_sept = subparsers.add_parser("sept")
    parser_sept.add_argument(
        "--sept_rows", action="store", type=int, default=1000, help="number of rows"
    )
    parser_sept.add_argument(
        "--sept_cols", action="store", type=int, default=10, help="number of cols"
    )

    parser_rept = subparsers.add_parser("rept")
    parser_rept.add_argument(
        "--rept_rows", action="store", type=int, default=1000, help="number of rows"
    )
    parser_rept.add_argument(
        "--rept_cols", action="store", type=int, default=10, help="number of cols"
    )
    parser_rept.add_argument(
        "--rept_dimension",
        action="store",
        type=int,
        default=15,
        help="dimension of the row tensor",
    )

    parser_all = subparsers.add_parser("all")
    parser_all.add_argument(
        "--sept_rows",
        action="store",
        type=int,
        default=1000,
        help="number of rows for the sept scenario",
    )
    parser_all.add_argument(
        "--sept_cols",
        action="store",
        type=int,
        default=10,
        help="number of cols for the sept scenario",
    )
    parser_all.add_argument(
        "--rept_rows",
        action="store",
        type=int,
        default=1000,
        help="number of rows for the rept scenario",
    )
    parser_all.add_argument(
        "--rept_cols",
        action="store",
        type=int,
        default=10,
        help="number of cols for the rept scenario",
    )
    parser_all.add_argument(
        "--rept_dimension",
        action="store",
        type=int,
        default=15,
        help="dimension of the row tensor",
    )

    return parser


def add_cmd_args(cmd, args):
    if args.run_type == "sept":
        cmd.append("sept")

    if args.run_type == "rept":
        cmd.append("rept")

    if args.run_type == "all":
        cmd.append("all")

    if args.run_type in ["sept", "all"]:
        cmd.append(f"--sept_rows={args.sept_rows}")
        cmd.append(f"--sept_cols={args.sept_cols}")

    if args.run_type in ["rept", "all"]:
        cmd.append(f"--rept_rows={args.rept_rows}")
        cmd.append(f"--rept_cols={args.rept_cols}")
        cmd.append(f"--rept_dimension={args.rept_dimension}")


def run_suite() -> None:
    inf = np.iinfo(np.int32)
    parser = get_parser()
    runner = pyperf.Runner(_argparser=parser, add_cmdline_args=add_cmd_args)
    runner.parse_args()

    runner.metadata["git_commit_hash"] = get_git_revision_short_hash()

    # run_phitensor_suite(runner=runner, data_file=data_file)
    args = runner.args
    # print(args)
    if args.run_type in ["sept", "all"]:
        run_sept_suite(
            runner=runner,
            rows=args.sept_rows,
            cols=args.sept_cols,
            lower_bound=inf.min,
            upper_bound=inf.max,
        )

    if args.run_type in ["rept", "all"]:
        run_rept_suite(
            runner=runner,
            rept_dimension=args.rept_dimension,
            rows=args.rept_rows,
            cols=args.rept_cols,
            lower_bound=inf.min,
            upper_bound=inf.max,
        )


if __name__ == "__main__":
    run_suite()
