"""The syft migration registry exists and computes a valid protocol schema."""

from syft.migrations import (
    PROTOCOL_NAME,
    SYFT_CLIENT_PROTOCOL_VERSION,
    client_registry,
)
from syft.version import SYFT_VERSION


def test_registry_identity():
    assert client_registry.protocol_name == PROTOCOL_NAME
    assert client_registry.package_name == "syft"
    assert client_registry.package_version == SYFT_VERSION
    assert client_registry.protocol_version == SYFT_CLIENT_PROTOCOL_VERSION


def test_registry_computes_protocol_schema():
    schema = client_registry.compute_protocol_schema()
    assert schema.protocol_name == PROTOCOL_NAME
    assert schema.version == SYFT_CLIENT_PROTOCOL_VERSION
    # Every registered object resolves a current version and a frozen schema.
    for canonical_name in schema.supported_versions:
        assert schema.current_schema(canonical_name=canonical_name)
        assert canonical_name in schema.current_object_schemas
