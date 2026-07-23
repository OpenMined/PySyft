"""The syft-client migration registry exists and computes a valid protocol schema."""

from syft_client.migrations import (
    PROTOCOL_NAME,
    SYFT_CLIENT_PROTOCOL_VERSION,
    client_registry,
)
from syft_client.version import SYFT_CLIENT_VERSION


def test_registry_identity():
    assert client_registry.protocol_name == PROTOCOL_NAME
    assert client_registry.package_name == "syft-client"
    assert client_registry.package_version == SYFT_CLIENT_VERSION
    assert client_registry.protocol_version == SYFT_CLIENT_PROTOCOL_VERSION


def test_registry_computes_empty_protocol_schema():
    # No versioned objects are registered yet (they arrive in later waves);
    # the registry must still compute a well-formed schema.
    schema = client_registry.compute_protocol_schema()
    assert schema.protocol_name == PROTOCOL_NAME
    assert schema.version == SYFT_CLIENT_PROTOCOL_VERSION
    assert schema.supported_versions == {}
    assert schema.current_object_schemas == {}
