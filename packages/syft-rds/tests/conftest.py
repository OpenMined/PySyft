"""Pytest configuration for the syft-rds test suite.

Mirrors the top-level ``tests/conftest.py`` so RDS product tests behave the same
whether run here or as part of the full monorepo suite.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_pre_sync_for_tests():
    """Disable PRE_SYNC by default for all tests (explicit sync control)."""
    original_value = os.environ.get("PRE_SYNC")
    os.environ["PRE_SYNC"] = "false"

    yield

    if original_value is not None:
        os.environ["PRE_SYNC"] = original_value
    else:
        os.environ.pop("PRE_SYNC", None)
