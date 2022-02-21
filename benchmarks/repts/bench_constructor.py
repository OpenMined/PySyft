import pyperf
from ..septs.util import make_sept, generate_data
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor as REPT
import functools

def create_bench_constructor(runner: pyperf.Runner, rept_length: int, rows:int, cols: int, lower_bound: int, upper_bound: int) -> None:
    rept_rows = []

    for i in range(rept_length):
        sept_data = generate_data(rows, cols, lower_bound, upper_bound)
        sept = make_sept(sept_data, lower_bound, upper_bound)
        rept_rows.append(sept)

    partial_function_evaluation = functools.partial(REPT, rept_rows)
    runner.bench_func("rept_creation_rept_length_{rept_length}_rows_{rows}_cols_{cols}", partial_function_evaluation)
