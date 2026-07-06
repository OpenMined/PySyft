"""The schema model classes themselves (no registry state)."""

import pytest

from syft_migration import (
    MigrationError,
    PackageInfo,
    ProtocolSchema,
    ReleaseArtifact,
)

from mocks import JobV1, JobV2


def test_schema_save_load_roundtrip(tmp_path):
    schema = ProtocolSchema.from_objects(
        protocol_name="mock-proto",
        version="2",
        classes=[JobV2],
    )
    path = tmp_path / "protocol.json"
    schema.save(path=path)
    loaded = ProtocolSchema.load(path=path)
    assert loaded.protocol_name == "mock-proto"
    assert loaded.version == "2"
    assert loaded.supported_versions == {"job": ["2"]}


def test_schema_collects_all_versions_of_same_object():
    schema = ProtocolSchema.from_objects(
        protocol_name="p",
        version="1",
        classes=[JobV2, JobV1],
    )
    assert schema.supported_versions == {"job": ["1", "2"]}
    assert schema.current_schema(canonical_name="job") == "2"


def test_current_schema_raises_on_unknown_object():
    schema = ProtocolSchema.from_objects(
        protocol_name="p",
        version="1",
        classes=[JobV1],
    )
    with pytest.raises(MigrationError):
        schema.current_schema(canonical_name="unknown")


def test_protocol_schema_save_load_roundtrip(tmp_path):
    protocol = ProtocolSchema(
        protocol_name="mock-proto",
        version="2",
        supported_versions={"job": ["1", "2"]},
    )
    path = tmp_path / "protocol_schema.json"
    protocol.save(path=path)
    assert ProtocolSchema.load(path=path) == protocol


def test_release_artifact_save_load_roundtrip(tmp_path):
    artifact = ReleaseArtifact(
        package_info=PackageInfo(
            package_name="syft-mock", version="1.0.0", protocol_version="1"
        ),
        protocol_schema=ProtocolSchema.from_objects(
            protocol_name="mock-proto",
            version="1",
            classes=[JobV1],
        ),
    )
    path = tmp_path / "release.json"
    artifact.save(path=path)
    assert ReleaseArtifact.load(path=path) == artifact
