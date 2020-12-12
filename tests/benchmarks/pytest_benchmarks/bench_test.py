"""
Define benchmark tests
"""
from ..pytest_benchmarks.benchmarks_functions_test import string_serde
from typing import Any


def test_string_serde(benchmark: Any) -> None:
    """
    Test sigmoid approximation with chebyshev method and
    precision value of 4
    """
    benchmark(string_serde)
