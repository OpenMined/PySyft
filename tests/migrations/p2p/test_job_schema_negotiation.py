"""Peer-advertised job schemas drive JobStorage protocol negotiation."""

from pathlib import Path

from syft_job import SyftJobConfig
from syft_job.client import JobClient
from syft_job.job_storage import JobStorage
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION
from syft_migration import ProtocolSchema

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def _job_schema(protocol_version: str) -> ProtocolSchema:
    # The slim form a peer advertises in its VersionInfo (see
    # version_info._slim_schema_of): no embedded object schemas.
    return ProtocolSchema(
        protocol_name="syft-job",
        version=protocol_version,
        supported_versions={"JobState": ["1"], "JobSubmissionMetadata": ["1"]},
    )


def _storage(tmp_path: Path, peer_schemas: dict) -> JobStorage:
    config = SyftJobConfig(
        syftbox_folder=tmp_path / "SyftBox", current_user_email=DS_EMAIL
    )
    (tmp_path / "SyftBox" / DS_EMAIL).mkdir(parents=True, exist_ok=True)
    return JobStorage(config=config, peer_schemas=peer_schemas)


def test_protocol0_peer_negotiates_down(tmp_path):
    storage = _storage(tmp_path, {DO_EMAIL: _job_schema("0")})
    assert storage.negotiated_protocol_version_for_peer(DO_EMAIL) == "0"
    ref = storage.new_submission_ref(DO_EMAIL, "legacy.job")
    assert ref.protocol_version == "0"
    # Protocol 0 = flat pre-versioning layout: no v<n> path segment.
    assert f"v{JOB_PROTOCOL_VERSION}" not in storage.submission_dir(ref).parts


def test_current_peer_negotiates_current(tmp_path):
    storage = _storage(tmp_path, {DO_EMAIL: _job_schema(JOB_PROTOCOL_VERSION)})
    assert (
        storage.negotiated_protocol_version_for_peer(DO_EMAIL) == JOB_PROTOCOL_VERSION
    )
    ref = storage.new_submission_ref(DO_EMAIL, "current.job")
    assert ref.protocol_version == JOB_PROTOCOL_VERSION
    assert f"v{JOB_PROTOCOL_VERSION}" in storage.submission_dir(ref).parts


def test_unknown_peer_keeps_current_protocol_assumption(tmp_path):
    storage = _storage(tmp_path, {})
    ref = storage.new_submission_ref(DO_EMAIL, "unknown.job")
    assert ref.protocol_version == JOB_PROTOCOL_VERSION


def test_live_map_updates_are_seen_by_storage(tmp_path):
    # JobStorage holds the dict by reference: schemas arriving after
    # construction (peer version files loading) change negotiation.
    live: dict = {}
    storage = _storage(tmp_path, live)
    assert (
        storage.new_submission_ref(DO_EMAIL, "before.job").protocol_version
        == JOB_PROTOCOL_VERSION
    )
    live[DO_EMAIL] = _job_schema("0")
    assert storage.new_submission_ref(DO_EMAIL, "after.job").protocol_version == "0"


def test_job_client_from_config_passes_schemas_through(tmp_path):
    config = SyftJobConfig(
        syftbox_folder=tmp_path / "SyftBox", current_user_email=DS_EMAIL
    )
    (tmp_path / "SyftBox" / DS_EMAIL).mkdir(parents=True, exist_ok=True)
    live = {DO_EMAIL: _job_schema("0")}
    client = JobClient.from_config(config, peer_schemas=live)
    assert client.manager.peer_schemas is live


def test_newer_peer_clamps_to_our_protocol(tmp_path):
    # A peer speaking a future protocol negotiates down to ours (min).
    storage = _storage(tmp_path, {DO_EMAIL: _job_schema("99")})
    assert (
        storage.negotiated_protocol_version_for_peer(DO_EMAIL) == JOB_PROTOCOL_VERSION
    )


def test_downgrade_write_uses_slim_peer_schema(tmp_path):
    # The write path's downgrade target comes from the slim advertised schema
    # (supported_versions only) — no dependency on current_object_schemas.
    from syft_job.models import JobSubmissionMetadataV1
    from datetime import datetime, timezone

    storage = _storage(tmp_path, {DO_EMAIL: _job_schema("0")})
    ref = storage.new_submission_ref(DO_EMAIL, "legacy.job")
    metadata = JobSubmissionMetadataV1(
        name="legacy.job",
        submitted_by=DS_EMAIL,
        datasite_email=DO_EMAIL,
        submitted_at=datetime.now(tz=timezone.utc),
    )
    path = storage.write_submission(ref, metadata)
    assert path.exists()
    assert storage.read_submission(ref).name == "legacy.job"
