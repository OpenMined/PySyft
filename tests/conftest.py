"""
Pytest configuration for all tests.

This file is automatically loaded by pytest and configures the test environment.
"""

import os
from typing import NamedTuple

import pytest
from syft_migration import protocol_version_at_least

from syft.migrations import PROTOCOL_NAME, client_registry
from syft.sync.peers.peer_store import PeerStore
from syft.sync.syftbox_manager import SyftboxManager


class ProtocolStep(NamedTuple):
    """The syft protocol of the last release, and the next one."""

    released: str
    next: str


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


@pytest.fixture
def next_protocol(monkeypatch) -> ProtocolStep:
    """A test-only next syft protocol, with the current one frozen as released.

    The registry stores its current schema as a released protocol, then speaks
    one version higher. A client that ``pin_syft_protocol`` holds at the
    released version then plays a client of the last release. Build the
    clients after this fixture, so they publish the next version.
    """
    released = client_registry.protocol_version
    monkeypatch.setitem(
        client_registry.protocol_version_history,
        released,
        client_registry.compute_protocol_schema(),
    )
    next_version = str(int(released) + 1)
    monkeypatch.setattr(client_registry, "protocol_version", next_version)
    return ProtocolStep(released=released, next=next_version)


@pytest.fixture
def pin_syft_protocol(monkeypatch, advertised_schemas):
    """Hold a client at an older syft protocol, as a client of an older release.

    ``pin_syft_protocol(manager, version)`` republishes the version file of
    ``manager`` with the schema of ``version``, and caps every peer's syft
    schema that ``manager`` reads at ``version``. Its own downgrade and switch
    then negotiate at most ``version``. Call ``load_peers()`` on its peers
    afterwards, so they read the new version file.

    It wraps ``advertised_schemas``, so an advertised schema is capped too.

    Only the negotiation is held back. Code that reads
    ``client_registry.protocol_version`` directly still acts at the next
    version, so gate a new behaviour on ``ConnectionRouter.pair_speaks_protocol()``.
    """
    # email of a held client -> the syft protocol version it is held at
    held_versions: dict[str, str] = {}
    read_schema = PeerStore.protocol_schema

    def protocol_schema(self, email, protocol_name):
        schema = read_schema(self, email, protocol_name)
        held = held_versions.get(self.email)
        if schema is None or held is None or protocol_name != PROTOCOL_NAME:
            return schema
        if protocol_version_at_least(held, schema.version):
            return schema
        held_schema = client_registry.schema_for_protocol_version(held)
        return held_schema.model_copy(
            update={"min_supported_version": schema.min_supported_version}
        )

    monkeypatch.setattr(PeerStore, "protocol_schema", protocol_schema)

    def pin(manager: SyftboxManager, version: str) -> None:
        peer_manager = manager.peer_manager
        own_version = peer_manager.get_own_version()
        held_schema = client_registry.schema_for_protocol_version(version)
        peer_manager._own_version = own_version.model_copy(
            update={
                "protocol_schemas": {
                    **own_version.protocol_schemas,
                    PROTOCOL_NAME: held_schema,
                }
            }
        )
        peer_manager.write_own_peer_info()
        held_versions[manager.email] = version

    return pin
