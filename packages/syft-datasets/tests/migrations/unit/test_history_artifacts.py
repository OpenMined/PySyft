"""The hardcoded release artifacts of past syft-dataset releases."""

from syft_datasets.migrations import dataset_registry
from syft_datasets.migrations.history import PACKAGE_ARTIFACTS_DIR, PROTOCOLS_DIR
from syft_datasets.models import DatasetV1
from syft_migration import (
    MigrationService,
    ReleasedPackageProtocolInfo,
    ReleasedProtocol,
)


def test_all_released_package_artifacts_load():
    artifact_paths = sorted(PACKAGE_ARTIFACTS_DIR.glob("*.json"))
    assert artifact_paths  # at least syft-dataset-0.1.20.json exists

    for path in artifact_paths:
        artifact = ReleasedPackageProtocolInfo.load(path)
        info = artifact.package_info
        # The filename encodes the package version: syft-dataset-<version>.json.
        assert path.name == f"{info.package_name}-{info.version}.json"
        assert info.package_name == "syft-dataset"
        schema = artifact.protocol_schema
        assert schema.protocol_name == "syft-dataset"
        assert schema.version == info.protocol_version
        assert schema.supported_versions
        assert set(schema.current_object_schemas) == set(schema.supported_versions)
        for canonical_name in schema.supported_versions:
            assert schema.current_schema(canonical_name)


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


def test_protocol_bump_not_missing():
    # Stays live between a protocol bump and the release that freezes it, which is
    # exactly where test_protocol_bumped_when_changed goes quiet.
    assert not dataset_registry.protocol_bump_missing(), (
        "The dataset protocol changed since the newest released protocol without a "
        "bump. Bump DATASET_PROTOCOL_VERSION in "
        "syft_datasets/migrations/registry.py, or revert the model change."
    )


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
