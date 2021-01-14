"""
Define benchmark tests
"""
# stdlib
from typing import Any

# third party
import pytest

# syft relative
from ..pytest_benchmarks.benchmarks_functions_test import string_serde


@pytest.mark.benchmark
def test_string_serde(benchmark: Any) -> None:
    """
    Test sigmoid approximation with chebyshev method and
    precision value of 4
    """
    benchmark(string_serde)
