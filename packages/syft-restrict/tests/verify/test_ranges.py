"""Tests for verify()'s private-range argument itself, not the code inside it."""

import pytest
from syft_restrict import verify

from .conftest import normalize_source


def test_inverted_range_must_raise_not_silently_pass(policy):
    # A malformed range (end < start) must never be mistaken for "nothing to
    # check here" -- it must raise, not silently verify zero nodes and report
    # ok=True.
    src = normalize_source("""
    import os
    def f(x):
        os.system("rm -rf /")
        return x
    """)
    with pytest.raises(ValueError):
        verify(src, [[4, 2]], policy)
