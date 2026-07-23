"""The first real migrations: VersionInfo v1 <-> v2 (protocol schemas)."""

import json
from pathlib import Path

from syft_client.migrations import client_migration_service, client_registry
from syft_client.sync.version.version_info import (
    VersionInfo,
    VersionInfoV1,
    VersionInfoV2,
)

LEGACY_FILE = (
    Path(__file__).parent / "fixtures" / "version_info" / "SYFT_version-0.1.117.json"
)


def _v1() -> VersionInfoV1:
    return VersionInfoV1.model_validate(json.loads(LEGACY_FILE.read_text()))


def test_both_versions_registered_with_paths_both_ways():
    assert client_registry.versions("VersionInfo") == ["1", "2"]
    assert client_registry.has_migration_path("VersionInfo", "1", "2")
    assert client_registry.has_migration_path("VersionInfo", "2", "1")


def test_v1_upgrades_to_v2_with_empty_schemas():
    upgraded = client_migration_service.migrate(_v1(), "2")
    assert type(upgraded) is VersionInfoV2
    assert upgraded.version == "2"
    # A v1 file says nothing about package protocols.
    assert upgraded.protocol_schemas == {}
    assert upgraded.syft_client_version == "0.1.117"
    assert upgraded.updated_at == _v1().updated_at


def test_v2_downgrades_to_v1_dropping_schemas():
    current = VersionInfo.current()
    assert current.protocol_schemas  # populated before the downgrade
    downgraded = client_migration_service.migrate(current, "1")
    assert type(downgraded) is VersionInfoV1
    assert downgraded.version == "1"
    assert "protocol_schemas" not in downgraded.model_dump()
    assert downgraded.syft_client_version == current.syft_client_version


def test_downgrade_for_protocol_0_peer():
    # A protocol-0 peer's schema only lists VersionInfo v1.
    downgraded = client_migration_service.downgrade_for_protocol_version(
        VersionInfo.current(), "0"
    )
    assert type(downgraded) is VersionInfoV1


def test_current_advertises_slim_schemas():
    schemas = VersionInfo.current().protocol_schemas
    # The client's own schema is always present; job/dataset only when the
    # optional packages are importable (both are in the workspace env, but
    # the code must degrade on client-only installs).
    assert "syft-client" in schemas
    assert set(schemas) <= {"syft-client", "syft-job", "syft-dataset"}
    client_schema = schemas["syft-client"]
    assert client_schema.version == client_registry.protocol_version
    assert client_schema.supported_versions == (
        client_registry.compute_protocol_schema().supported_versions
    )
    # Slim on the wire: no embedded per-object JSON schemas.
    for schema in schemas.values():
        assert schema.current_object_schemas == {}


def test_legacy_file_loads_all_the_way_to_v2():
    info = VersionInfo.from_json(LEGACY_FILE.read_text())
    assert type(info) is VersionInfoV2
    assert info.protocol_schemas == {}
