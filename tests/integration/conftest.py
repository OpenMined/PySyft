"""
Pytest fixtures for integration tests.

Fixtures defined here are automatically available to all tests in this directory.
"""

import os

import pytest

from tests.integration.utils import (
    token_path_do,
    token_path_ds,
    remove_syftboxes_from_drive,
)


@pytest.fixture()
def setup_delete_syftboxes():
    """Clean up syftboxes from drive before running integration tests."""
    if os.environ.get("INTEGRATION_TEST_MOCK_MODE", "").lower() == "true":
        yield
        return

    tokens_exist = token_path_do.exists() and token_path_ds.exists()
    if not tokens_exist:
        raise ValueError(
            """Credentials not found, create them using scripts/create_token.py and store them in /credentials
            as token_do.json and token_ds.json. Also set the environment variables AI_AUDIT_EMAIL_DO and AI_AUDIT_EMAIL_DS to the email addresses of the DO and DS."""
        )
    remove_syftboxes_from_drive()
    yield
