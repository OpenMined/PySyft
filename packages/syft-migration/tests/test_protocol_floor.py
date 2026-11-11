"""A protocol floor refuses a version that one of the two sides cannot read.

Both sides publish a floor. The lower of the two current versions is chosen, and
that version must be at or above both floors. A floor of "0" refuses nothing.
"""

import pytest
from syft_migration import MigrationError, MigrationRegistry, ProtocolSchema


def _registry(protocol_version: str = "2", floor: str = "0") -> MigrationRegistry:
    return MigrationRegistry(
        protocol_name="p",
        package_name="pkg",
        package_version="1.0.0",
        protocol_version=protocol_version,
        min_supported_protocol_version=floor,
    )


def _peer_schema(version: str, floor: str = "0") -> ProtocolSchema:
    """The schema a peer advertises."""
    return ProtocolSchema(
        protocol_name="p", version=version, min_supported_version=floor
    )


def test_schema_floor_defaults_to_zero():
    # A peer that predates the floor field says nothing, so it refuses nothing.
    schema = ProtocolSchema(protocol_name="p", version="1")
    assert schema.min_supported_version == "0"


def test_registry_floor_defaults_to_zero():
    reg = MigrationRegistry(
        protocol_name="p",
        package_name="pkg",
        package_version="1.0.0",
        protocol_version="1",
    )
    assert reg.min_supported_protocol_version == "0"


def test_lower_version_chosen():
    reg = _registry(protocol_version="2")
    assert reg.infer_protocol_version_to_use(_peer_schema("1")) == "1"
    assert reg.infer_protocol_version_to_use(_peer_schema("3")) == "2"


def test_versions_order_by_number():
    reg = _registry(protocol_version="10")
    assert reg.infer_protocol_version_to_use(_peer_schema("9")) == "9"


def test_our_floor_refuses_older_peer():
    reg = _registry(protocol_version="2", floor="2")
    with pytest.raises(MigrationError, match="1"):
        reg.infer_protocol_version_to_use(_peer_schema("1"))


def test_peer_floor_refuses_us():
    reg = _registry(protocol_version="2", floor="0")
    with pytest.raises(MigrationError):
        reg.infer_protocol_version_to_use(_peer_schema("3", floor="3"))


def test_zero_floor_on_both_sides_refuses_nothing():
    reg = _registry(protocol_version="5", floor="0")
    assert reg.infer_protocol_version_to_use(_peer_schema("0", floor="0")) == "0"
