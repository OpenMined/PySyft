"""Release artifacts register cleanly and guard against schema drift.

The artifacts under syft/migrations/history are written by the release job, so
a protocol that has not been released yet has none. Each guard below holds with
or without them.
"""

import pytest
import syft  # noqa: F401 -- imports models and registers history
from syft_migration import MigrationError, MigrationRegistry, ReleasedProtocol

from syft.migrations import client_registry
from syft.migrations.history import PROTOCOLS_DIR, register_historic_schemas


def _fresh_registry(protocol_version: str) -> MigrationRegistry:
    return MigrationRegistry(
        protocol_name=client_registry.protocol_name,
        package_name=client_registry.package_name,
        package_version=client_registry.package_version,
        protocol_version=protocol_version,
    )


def test_registering_again_is_idempotent():
    before = dict(client_registry.protocol_version_history)
    register_historic_schemas()
    assert client_registry.protocol_version_history == before


def test_all_protocol_artifacts_well_formed():
    # Filename encodes the frozen protocol version, and every supported
    # canonical name freezes a current-object schema (catches a mis-named or
    # hand-edited artifact).
    for path in sorted(PROTOCOLS_DIR.glob("*.json")):
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
        "SYFT_CLIENT_PROTOCOL_VERSION in syft/migrations/registry.py. "
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
        "syft/migrations/registry.py, or revert the model change."
    )


def test_bump_guard_trips_on_protocol_change():
    # A registry claiming the same protocol version as a released schema but
    # supporting different object versions must trip the guard.
    released = client_registry.compute_protocol_schema()
    stale = _fresh_registry(released.version)
    # Register only a subset of the released objects, then load its schema.
    stale.register_object_version(
        client_registry.get_class(
            "VersionInfo", client_registry.latest_version("VersionInfo")
        )
    )
    stale.register_historic_protocol_schema(released)
    assert stale.protocol_changed_without_bump()


def test_unknown_object_version_in_artifact_raises():
    # The fail-at-import guarantee syft/__init__.py relies on: an
    # artifact naming an object version this release cannot load must raise.
    current = client_registry.latest_version("VersionInfo")
    schema = client_registry.compute_protocol_schema()
    schema = schema.model_copy(
        update={
            "supported_versions": {
                **schema.supported_versions,
                "VersionInfo": [current, "99"],
            }
        }
    )
    empty = _fresh_registry(client_registry.protocol_version)
    empty.register_object_version(client_registry.get_class("VersionInfo", current))
    with pytest.raises(MigrationError):
        empty.register_historic_protocol_schema(schema, raise_for_unknown_objects=True)
