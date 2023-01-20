"""
This file can be used to parametrize the benchmarking of autodp tensor operations
This file can be run as:
python executor.py MODE_OF_OPERATION SUITE_ARGS PERF_ARGS

Parameters
----------
SUITE_ARGS:
    These arguments control the size of the generated data used in our tests
PERF_ARGS:
    These arguments are inherited from the pyperf runner class,
    for more info: https://pyperf.readthedocs.io/en/latest/runner.html

"""
# future
from __future__ import annotations

# stdlib
import argparse
import inspect
import subprocess

# third party
from data import get_data_size
import pyperf
import syft_benchmarks


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def new_get_parser(params):
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument(
        "--select_tests",
        nargs="*",
        help="TODO",
        choices=list(params.keys()),
        action="store",
        required=True,
    )

    for test_name in params:
        for arg_name in params[test_name]:
            parser.add_argument(
                f"--{test_name}_{arg_name}",
                action="store",
                type=params[test_name][arg_name],
                help="TODO",
            )

    return parser


def get_tests_parameters():
    params = {}
    for method in dir(syft_benchmarks):
        if method[:4] == "run_" and method[-6:] == "_suite":
            suite_name = method[4:-6]
            params[suite_name] = {}
            run_func = getattr(syft_benchmarks, method)
            annotations = inspect.getfullargspec(run_func).annotations
            for k in annotations:
                if k != "runner" and k != "return":
                    params[suite_name][k] = annotations[k]

    return params


def new_add_cmd_args(cmd, args):
    if len(args.select_tests) > 0:
        cmd.append("--select_tests")
        for test_name in args.select_tests:
            cmd.append(test_name)
            for name_attr in dir(args):
                if name_attr.startswith(test_name):
                    value = getattr(args, name_attr)
                    if value:
                        cmd.append(f"--{name_attr}")
                        cmd.append(str(value))


def new_run_suite() -> None:
    # parse the modules
    params = get_tests_parameters()

    # create the base parser
    parser = new_get_parser(params)

    # setup the runner
    runner = pyperf.Runner(_argparser=parser, add_cmdline_args=new_add_cmd_args)
    runner.parse_args()

    runner.metadata["git_commit_hash"] = get_git_revision_short_hash()

    key = "100K"
    data_file, key = get_data_size(key)

    args = runner.args
    print(args)
    for test_name in args.select_tests:
        method_name = f"run_{test_name}_suite"
        run_func = getattr(syft_benchmarks, method_name)
        kwargs_dict = {"runner": runner, "data_file": data_file}
        for arg_name in params[test_name]:
            value = getattr(runner.args, f"{test_name}_{arg_name}")
            if value:
                kwargs_dict[arg_name] = value
        run_func(**kwargs_dict)


if __name__ == "__main__":
    new_run_suite()
