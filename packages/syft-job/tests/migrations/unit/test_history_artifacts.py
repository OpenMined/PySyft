"""The release artifacts of syft-job releases guard against schema drift.

The artifacts under syft_job/migrations/history are written by the release job,
so a protocol that has not been released yet has none. Each guard below holds
with or without them.
"""

from syft_migration import ReleasedProtocol

from syft_job.migrations import job_registry
from syft_job.migrations.history import PROTOCOLS_DIR


def test_all_released_protocols_load():
    for path in sorted(PROTOCOLS_DIR.glob("*.json")):
        released = ReleasedProtocol.load(path)
        schema = released.protocol_schema
        # The filename encodes the protocol version: protocol-<n>.json.
        assert path.name == f"protocol-{schema.version}.json"
        assert schema.protocol_name == "syft-job"
        # Every released protocol freezes the schema of each object's current
        # version, and lists that version as supported.
        assert schema.supported_versions
        assert set(schema.current_object_schemas) == set(schema.supported_versions)
        for canonical_name in schema.supported_versions:
            assert schema.current_schema(canonical_name)


def test_released_object_schemas_unchanged():
    # Released object versions are frozen forever; see the failure message.
    drift = job_registry.find_schema_drift()
    assert drift == [], (
        f"Released object schemas changed: {drift} "
        "(canonical_name, object_version, protocol_version).\n"
        "A class that shipped in a released protocol was modified in place. "
        "Released versions are frozen forever, because peers on old releases "
        "still read/write them. To fix:\n"
        "  1. Revert your change to the released class (e.g. JobStateV0).\n"
        "  2. Create the next version instead: copy the class into "
        "models/<object>/v<x+1>.py with version='<x+1>' and apply your change "
        "there (copy changed nested objects like enums into the new file too).\n"
        "  3. Point the current-version alias in models/<object>/__init__.py "
        "at the new class.\n"
        "  4. Register migrations in BOTH directions (v<x> -> v<x+1> and back) "
        "so old peers stay supported; test_upgrade_paths enforces this.\n"
        "  5. Add a serialized fixture: tests/migrations/unit/fixtures/"
        "<CanonicalName>/v<x+1>.yaml.\n"
        "  6. Bump JOB_PROTOCOL_VERSION in syft_job/migrations/registry.py — a "
        "new object version is a protocol change.\n"
        "If NO class was edited and this failure appeared after a pydantic "
        "upgrade, model_json_schema() output changed cosmetically: review the "
        "diff carefully and regenerate the files under migrations/history/."
    )


def test_protocol_bumped_when_changed():
    # Fires when the job protocol changes (e.g. a new object version registers)
    # without bumping JOB_PROTOCOL_VERSION.
    assert not job_registry.protocol_changed_without_bump()


def test_protocol_bump_not_missing():
    # Stays live between a protocol bump and the release that freezes it, which is
    # exactly where test_protocol_bumped_when_changed goes quiet.
    assert not job_registry.protocol_bump_missing(), (
        "The job protocol changed since the newest released protocol without a "
        "bump. Bump JOB_PROTOCOL_VERSION in syft_job/migrations/registry.py, or "
        "revert the model change."
    )
