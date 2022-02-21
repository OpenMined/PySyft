from ..septs.util import generate_data, make_sept
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor as REPT


def make_rept(rept_length: int, rows: int, cols: int, lower_bound: int, upper_bound: int) -> REPT:
    rept_rows = []

    for i in range(rept_length):
        sept_data = generate_data(rows, cols, lower_bound, upper_bound)
        sept = make_sept(sept_data, lower_bound, upper_bound)
        rept_rows.append(sept)

    return REPT(rept_rows)
