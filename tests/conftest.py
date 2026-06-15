"""
Pytest configuration for all tests.

This file is automatically loaded by pytest and configures the test environment.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_pre_sync_for_tests():
    """
    Disable PRE_SYNC by default for all tests.

    PRE_SYNC is enabled by default in production, but for tests we want
    explicit control over when sync() is called to avoid unexpected behavior
    and make tests faster.

    Individual tests can override this by setting os.environ["PRE_SYNC"] = "true"
    if they specifically want to test the auto-sync behavior.
    """
    original_value = os.environ.get("PRE_SYNC")
    os.environ["PRE_SYNC"] = "false"

    yield

    # Restore original value after all tests
    if original_value is not None:
        os.environ["PRE_SYNC"] = original_value
    else:
        os.environ.pop("PRE_SYNC", None)
