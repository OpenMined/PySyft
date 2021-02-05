"""
Define benchmark tests
"""
# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy

# syft relative
from ..pytest_benchmarks.benchmark_send_get_strings_local_test import (
    send_get_string_local,
)
from ..pytest_benchmarks.benchmarks_functions_test import string_serde


@pytest.mark.benchmark
def test_string_serde(benchmark: Any) -> None:
    """
    Test sigmoid approximation with chebyshev method and
    precision value of 4
    """
    benchmark(string_serde)


@pytest.mark.benchmark
@pytest.mark.parametrize("byte_size", [2 ** 10, 2 ** 20, 50 * 2 ** 20])
def test_duet_big_string_local(byte_size: int, benchmark: Any) -> None:
    data = "a" * byte_size
    duet = sy.VirtualMachine().get_root_client()

    benchmark.pedantic(send_get_string_local, args=(data, duet))
