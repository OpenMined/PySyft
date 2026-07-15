"""The hardcoded release artifacts of past syft-dataset releases."""

from syft_migration import (
    MigrationService,
    ReleasedPackageProtocolInfo,
    ReleasedProtocol,
)

from syft_datasets.migrations import dataset_registry
from syft_datasets.migrations.history import PACKAGE_ARTIFACTS_DIR, PROTOCOLS_DIR
from syft_datasets.models import DatasetV1


def test_0_1_20_artifact_file_loads():
    artifact = ReleasedPackageProtocolInfo.load(
        PACKAGE_ARTIFACTS_DIR / "syft-dataset-0.1.20.json"
    )
    assert artifact.package_info.package_name == "syft-dataset"
    assert artifact.package_info.version == "0.1.20"
    assert artifact.package_info.protocol_version == "0"
    assert artifact.protocol_schema.protocol_name == "syft-dataset"
    assert artifact.protocol_schema.version == "0"
    expected_versions = {"Dataset": ["1"], "PrivateDatasetConfig": ["1"]}
    assert artifact.protocol_schema.supported_versions == expected_versions


def test_all_released_protocols_load():
    protocol_paths = sorted(PROTOCOLS_DIR.glob("*.json"))
    assert protocol_paths  # at least protocol-0.json exists

    for path in protocol_paths:
        released = ReleasedProtocol.load(path)
        schema = released.protocol_schema
        # The filename encodes the protocol version: protocol-<n>.json.
        assert path.name == f"protocol-{schema.version}.json"
        assert schema.protocol_name == "syft-dataset"
        assert schema.supported_versions
        assert set(schema.current_object_schemas) == set(schema.supported_versions)
        for canonical_name in schema.supported_versions:
            assert schema.current_schema(canonical_name)


def test_protocol_0_released_protocol_loads():
    released = ReleasedProtocol.load(PROTOCOLS_DIR / "protocol-0.json")
    schema = released.protocol_schema
    assert schema.version == "0"
    assert set(schema.current_object_schemas) == {"Dataset", "PrivateDatasetConfig"}
    assert schema.current_schema("Dataset") == "1"
    assert schema.current_schema("PrivateDatasetConfig") == "1"


def test_released_object_schemas_unchanged():
    # Released object versions are frozen forever; see the failure message.
    drift = dataset_registry.find_schema_drift()
    assert drift == [], (
        f"Released object schemas changed: {drift} "
        "(canonical_name, object_version, protocol_version).\n"
        "A class that shipped in a released protocol was modified in place. "
        "Released versions are frozen forever, because peers on old releases "
        "still read/write them. To fix:\n"
        "  1. Revert your change to the released class (e.g. DatasetV1).\n"
        "  2. Create the next version instead: copy the class into "
        "models/<object>/v<x+1>.py with version='<x+1>' and apply your change "
        "there.\n"
        "  3. Point the current-version alias in models/<object>/__init__.py "
        "at the new class.\n"
        "  4. Register migrations in BOTH directions (v<x> -> v<x+1> and back) "
        "so old peers stay supported; test_upgrade_paths enforces this.\n"
        "  5. Add a serialized fixture: tests/migrations/unit/fixtures/"
        "<CanonicalName>/v<x+1>.yaml.\n"
        "  6. Bump DATASET_PROTOCOL_VERSION in syft_datasets/migrations/"
        "registry.py — a new object version is a protocol change.\n"
        "If NO class was edited and this failure appeared after a pydantic "
        "upgrade, model_json_schema() output changed cosmetically: review the "
        "diff carefully and regenerate the files under migrations/history/."
    )


def test_protocol_bumped_when_changed():
    # Fires when the dataset protocol changes (e.g. a new object version registers)
    # without bumping DATASET_PROTOCOL_VERSION.
    assert not dataset_registry.protocol_changed_without_bump()


def test_historic_schemas_registered_on_import():
    # syft_datasets/__init__ registers every artifact in migrations/history/.
    assert dataset_registry.package_version_history["0"].version == "0.1.20"
    schema = dataset_registry.schema_for_protocol_version("0")
    assert schema.current_schema("Dataset") == "1"
    assert schema.current_schema("PrivateDatasetConfig") == "1"


def test_downgrade_for_last_released_protocol_version():
    service = MigrationService(registry=dataset_registry)
    from .mocks import create_mock_dataset

    dataset = create_mock_dataset()
    downgraded = service.downgrade_for_protocol_version(dataset, "0")
    assert isinstance(downgraded, DatasetV1)
    assert downgraded.name == "demo"
