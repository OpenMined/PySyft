"""Past release artifacts register cleanly and guard against schema drift."""

import pytest
import syft_client  # noqa: F401 -- imports models and registers history
from syft_migration import MigrationError, MigrationRegistry, ReleasedProtocol

from syft_client.migrations import client_registry
from syft_client.migrations.history import (
    PACKAGE_ARTIFACTS_DIR,
    PROTOCOLS_DIR,
    register_historic_schemas,
)

PROTOCOL_0_PATH = PROTOCOLS_DIR / "protocol-0.json"


def test_protocol0_artifacts_registered():
    # importing syft_client registered the hardcoded 0.1.117 artifacts.
    assert "0" in client_registry.protocol_version_history
    package_info = client_registry.package_version_history["0"]
    assert package_info.package_name == "syft-client"
    assert package_info.version == "0.1.117"

    schema = client_registry.protocol_version_history["0"]
    assert schema.supported_versions == {
        "VersionInfo": ["1"],
        "ProposedFileChangesMessage": ["1"],
        "FileChangeEventsMessage": ["1"],
    }


def test_registering_again_is_idempotent():
    register_historic_schemas()
    assert "0" in client_registry.protocol_version_history


def test_all_protocol_artifacts_well_formed():
    # Filename encodes the frozen protocol version, and every supported
    # canonical name freezes a current-object schema (catches a mis-named or
    # hand-edited artifact).
    paths = sorted(PROTOCOLS_DIR.glob("*.json"))
    assert paths, "no released protocol artifacts found"
    for path in paths:
        schema = ReleasedProtocol.load(path).protocol_schema
        assert path.name == f"protocol-{schema.version}.json"
        assert set(schema.current_object_schemas) == set(schema.supported_versions)


def test_no_schema_drift_against_released_protocols():
    # Every schema frozen by a released protocol must still be produced
    # byte-identically by the class registered for that version.
    assert client_registry.find_schema_drift() == [], (
        "A released object schema drifted. Fix by either: (1) reverting the "
        "model change; or (2) adding a new V<n+1> model class, registering "
        "migrations in both directions, and bumping "
        "SYFT_CLIENT_PROTOCOL_VERSION in syft_client/migrations/registry.py. "
        "If the drift comes from a pydantic upgrade changing JSON-schema "
        "output only, regenerate the artifacts instead."
    )


def test_protocol_not_changed_without_bump():
    assert not client_registry.protocol_changed_without_bump()


def test_protocol_bump_not_missing():
    # Stays live between a protocol bump and the release that freezes it, which is
    # exactly where test_protocol_not_changed_without_bump goes quiet.
    assert not client_registry.protocol_bump_missing(), (
        "The client protocol changed since the newest released protocol without a "
        "bump. Bump SYFT_CLIENT_PROTOCOL_VERSION in "
        "syft_client/migrations/registry.py, or revert the model change."
    )


def test_bump_guard_trips_on_protocol_change():
    # A registry claiming the same protocol version as a released schema but
    # supporting different object versions must trip the guard.
    stale = MigrationRegistry(
        protocol_name=client_registry.protocol_name,
        package_name=client_registry.package_name,
        package_version=client_registry.package_version,
        protocol_version="0",  # pretend we still speak the released protocol 0
    )
    # Register only a subset of protocol-0's objects, then load its schema.
    stale.register_object_version(client_registry.get_class("VersionInfo", "1"))
    stale.register_historic_protocol_schema(
        ReleasedProtocol.load(PROTOCOL_0_PATH).protocol_schema
    )
    assert stale.protocol_changed_without_bump()


def test_unknown_object_version_in_artifact_raises():
    # The fail-at-import guarantee syft_client/__init__.py relies on: an
    # artifact naming an object version this release cannot load must raise.
    schema = ReleasedProtocol.load(PROTOCOL_0_PATH).protocol_schema
    schema = schema.model_copy(
        update={
            "supported_versions": {
                **schema.supported_versions,
                "VersionInfo": ["1", "99"],
            }
        }
    )
    empty = MigrationRegistry(
        protocol_name=client_registry.protocol_name,
        package_name=client_registry.package_name,
        package_version=client_registry.package_version,
        protocol_version=client_registry.protocol_version,
    )
    empty.register_object_version(client_registry.get_class("VersionInfo", "1"))
    with pytest.raises(MigrationError):
        empty.register_historic_protocol_schema(schema, raise_for_unknown_objects=True)


def test_artifact_files_exist():
    assert (PACKAGE_ARTIFACTS_DIR / "syft-client-0.1.117.json").exists()
    assert PROTOCOL_0_PATH.exists()
