import pyperf
# syft absolute
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor as REPT
import functools

# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT


import subprocess

def generate_data(
    rows: int, columns: int, lower_bound: int, upper_bound: int
) -> np.array:
    return np.random.randint(
        lower_bound, upper_bound, size=(rows, columns), dtype=np.int32
    )


def generate_entity() -> Entity:
    return Entity(name="Ishan")


def make_bounds(data, bound: int) -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly
    generated b/w 0-1"""
    return np.ones_like(data) * bound


def make_sept(np_data: np.array, upper_bound, lower_bound) -> SEPT:
    return SEPT(
        child=np_data,
        entity=generate_entity(),
        max_vals=make_bounds(np_data, upper_bound),
        min_vals=make_bounds(np_data, lower_bound),
    )

def create_bench_rept_constructor(
    runner: pyperf.Runner,
    rept_length: int,
    rows: int,
    cols: int,
    lower_bound: int,
    upper_bound: int,
) -> None:
    rept_rows = []

    for i in range(rept_length):
        sept_data = generate_data(rows, cols, lower_bound, upper_bound)
        sept = make_sept(sept_data, lower_bound, upper_bound)
        rept_rows.append(sept)

    partial_function_evaluation = functools.partial(REPT, rept_rows)
    runner.bench_func(
        f"constructor_rept_rept_length_{rept_length}_rows_{rows}_cols_{cols}",
        partial_function_evaluation,
    )


def create_bench_rept_serialize(
    runner: pyperf.Runner,
    rept_dimension: int,
    rows: int,
    columns: int,
    lower_bound: int,
    upper_bound: int,
):
    rept = make_rept(rept_dimension, rows, columns, lower_bound, upper_bound)
    partially_evaluated_func = functools.partial(sy.serialize, rept, True)
    runner.bench_func(
        f"serialize_rept__rept_dimension_{rept_dimension}_rows_{rows}_columns_{columns}",
        partially_evaluated_func,
    )

def create_bench_rept_deserialize(
    runner: pyperf.Runner,
    rept_dimension: int,
    rows: int,
    columns: int,
    lower_bound: int,
    upper_bound: int,
):
    rept = make_rept(rept_dimension, rows, columns, lower_bound, upper_bound)
    serialized_data = sy.serialize(rept, to_bytes=True)
    partially_evaluated_func = functools.partial(
        sy.deserialize, serialized_data, from_bytes=True
    )
    runner.bench_func(
        f"deserialize_rept__rept_dimension_{rept_dimension}_rows_{rows}_columns_{columns}",
        partially_evaluated_func,
    )
    
def run_rept_suite(
    runner: pyperf.Runner,
    rept_dimension: int,
    rows: int,
    cols: int,
    lower_bound: int,
    upper_bound: int,
) -> None:
    create_bench_rept_constructor(
        runner, rept_dimension, rows, cols, lower_bound, upper_bound
    )
    create_bench_rept_serialize(
        runner, rept_dimension, rows, cols, lower_bound, upper_bound
    )
    create_bench_rept_deserialize(
        runner, rept_dimension, rows, cols, lower_bound, upper_bound
    )


def make_rept(
    rept_length: int, rows: int, cols: int, lower_bound: int, upper_bound: int
) -> REPT:
    rept_rows = []

    for i in range(rept_length):
        sept_data = generate_data(rows, cols, lower_bound, upper_bound)
        sept = make_sept(sept_data, lower_bound, upper_bound)
        rept_rows.append(sept)

    return REPT(rept_rows)



def create_bench_sept_constructor(
    runner: pyperf.Runner, rows: int, cols: int, lower_bound: int, upper_bound: int
) -> None:
    data = np.random.randint(
        lower_bound, upper_bound, size=(rows, cols), dtype=np.int32
    )
    partially_evaluated_func = functools.partial(
        make_sept, data, lower_bound, upper_bound
    )
    runner.bench_func(
        f"constructor_sept_rows_{rows}_cols_{cols}", partially_evaluated_func
    )

def create_bench_sept_deserialize(
    runner: pyperf.Runner, rows: int, columns: int, lower_bound: int, upper_bound: int
) -> None:
    data = generate_data(rows, columns, lower_bound, upper_bound)
    serialized_sept = sy.serialize(
        make_sept(data, lower_bound, upper_bound), to_bytes=True
    )
    partialy_evaluated_func = functools.partial(
        sy.deserialize, serialized_sept, from_bytes=True
    )
    runner.bench_func(
        f"deserialize_sept_rows_{rows}_columns_{columns}", partialy_evaluated_func
    )

def create_bench_sept_serialize(
    runner: pyperf.Runner, rows: int, columns: int, lower_bound: int, upper_bound: int
):
    data = generate_data(rows, columns, lower_bound, upper_bound)
    sept = make_sept(data, lower_bound, upper_bound)
    partially_evaluated_func = functools.partial(sy.serialize, sept, True)
    runner.bench_func(
        f"serialize_sept_rows_{rows}_columns_{columns}", partially_evaluated_func
    )


def run_sept_suite(runner: pyperf.Runner, rows, cols, lower_bound, upper_bound):
    create_bench_sept_deserialize(runner, rows, cols, lower_bound, upper_bound)
    create_bench_sept_constructor(runner, rows, cols, lower_bound, upper_bound)
    create_bench_sept_serialize(runner, rows, cols, lower_bound, upper_bound)

def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

# def modify_args(cmd, args):
    # print(cmd)
    # cmd.append("--runsept")
    # print(args)
    # return cmd

def run_suite() -> None:
    # print(sys.argv)
    # print(kwargs)
    inf = np.iinfo(np.int32)
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--runsept')
    runner = pyperf.Runner()
    # print(sys.argv)
    # runner.parse_args()
    # print(runner.args)
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


if __name__ == "__main__":
    run_suite()