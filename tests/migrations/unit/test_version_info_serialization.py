"""VersionInfo round-trips through JSON, and a file of another version is refused."""

import json

import pytest
from syft.migrations import client_registry
from syft.sync.version.version_info import VersionInfo, VersionInfoV0
from syft_migration import MigrationError


def test_version_info_registered_and_aliased():
    assert client_registry.versions("VersionInfo")
    assert VersionInfo is VersionInfoV0

    schema = client_registry.compute_protocol_schema()
    assert "VersionInfo" in schema.supported_versions
    assert schema.current_schema(canonical_name="VersionInfo")


def test_current_serializes_identity_fields():
    data = json.loads(VersionInfo.current().to_json())
    assert data["canonical_name"] == "VersionInfo"
    assert data["version"] == "0"


def test_json_round_trip():
    original = VersionInfo.current()
    restored = VersionInfo.from_json(original.to_json())
    assert restored == original


def test_a_file_without_identity_fields_is_refused():
    data = json.loads(VersionInfo.current().to_json())
    del data["canonical_name"], data["version"]
    with pytest.raises(MigrationError):
        VersionInfo.from_json(json.dumps(data))


def test_a_file_of_an_unknown_version_is_refused():
    data = json.loads(VersionInfo.current().to_json())
    data["version"] = "99"
    with pytest.raises(MigrationError):
        VersionInfo.from_json(json.dumps(data))
