"""
Pytest configuration for all tests.

This file is automatically loaded by pytest and configures the test environment.
"""

import os

import pytest

from syft.sync.peers.peer_store import PeerStore


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


@pytest.fixture
def advertised_schemas(monkeypatch):
    """Protocol schemas the test makes peers advertise.

    ``advertised_schemas[email, protocol_name] = schema`` replaces what the
    peer info of ``email`` says, for every client in the test. ``None`` makes
    the schema unknown. Use it to stand in for a peer on an older release.
    """
    overrides = {}
    read_from_peer_info = PeerStore.protocol_schema

    def protocol_schema(self, email, protocol_name):
        if (email, protocol_name) in overrides:
            return overrides[email, protocol_name]
        return read_from_peer_info(self, email, protocol_name)

    monkeypatch.setattr(PeerStore, "protocol_schema", protocol_schema)
    return overrides
