import numpy as np
import pyperf

from .util import make_sept

from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT


def create_bench_sept_constructor(runner: pyperf.Runner, rows: int, cols: int, lower_bound, upper_bound) -> SEPT:
    data = np.random.randint(lower_bound, upper_bound, size=(rows, cols), dtype=np.int32)
    runner.bench_func(f"sept_creation_rows_{rows}_cols_{cols}", make_sept, data, lower_bound, upper_bound)