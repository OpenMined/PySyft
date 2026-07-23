"""VersionInfo round-trips through JSON and legacy protocol-0 files still load."""

import json
from pathlib import Path

from syft_client.migrations import client_registry
from syft_client.sync.version.version_info import VersionInfo, VersionInfoV1

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "version_info"
LEGACY_FILE = FIXTURES_DIR / "SYFT_version-0.1.117.json"


def test_version_info_registered_and_aliased():
    assert client_registry.versions("VersionInfo")
    assert VersionInfo is VersionInfoV1

    schema = client_registry.compute_protocol_schema()
    assert "VersionInfo" in schema.supported_versions
    assert schema.current_schema(canonical_name="VersionInfo")


def test_current_serializes_identity_fields():
    data = json.loads(VersionInfo.current().to_json())
    assert data["canonical_name"] == "VersionInfo"
    assert data["version"] == "1"


def test_json_round_trip():
    original = VersionInfo.current()
    restored = VersionInfo.from_json(original.to_json())
    assert restored == original


def test_legacy_protocol0_file_loads_as_latest():
    # Written by a <= 0.1.117 client: no canonical_name/version fields.
    legacy_json = LEGACY_FILE.read_text()
    assert "canonical_name" not in json.loads(legacy_json)

    info = VersionInfo.from_json(legacy_json)
    assert isinstance(info, VersionInfoV1)
    assert info.version == client_registry.latest_version("VersionInfo")
    assert info.syft_client_version == "0.1.117"
    assert info.syft_client_install_source == "pip"


def test_legacy_reader_tolerates_identity_fields():
    # A protocol-0 client parses with pydantic's default extra="ignore"; the
    # closest stand-in we have is validating minus the identity defaults.
    data = json.loads(VersionInfo.current().to_json())
    # Legacy clients see unknown keys and ignore them; simulate by checking
    # the payload minus identity fields is exactly the legacy shape.
    data.pop("canonical_name")
    data.pop("version")
    assert set(data) == set(json.loads(LEGACY_FILE.read_text()))
