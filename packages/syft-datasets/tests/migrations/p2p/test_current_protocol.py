"""End-to-end dataset flow through SyftDatasetManager.

Datasets are broadcast (one public copy read by the whole audience), so by
default they are written in the widest-compatible (oldest) protocol so every
current peer can read them; the new v<n> layout is written only for peers that
advertise support for it.
"""

from pathlib import Path

import yaml
from syft_datasets.dataset_manager import SyftDatasetManager
from syft_datasets.migrations import dataset_registry

DO_EMAIL = "do@test.org"
DS0 = "old@test.org"
DS1 = "new@test.org"


def _create_dataset_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    src = tmp_path / "src"
    src.mkdir()
    mock = src / "mock.csv"
    mock.write_text("id,value\n1,10\n")
    private = src / "private.csv"
    private.write_text("id,secret\n1,x\n")
    readme = src / "readme.md"
    readme.write_text("# demo\n")
    return mock, private, readme


def _dataset_manager(tmp_path: Path, peer_schemas=None) -> SyftDatasetManager:
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    mgr = SyftDatasetManager(syftbox_folder_path=syftbox, email=DO_EMAIL)
    if peer_schemas is not None:
        mgr.storage.peer_schemas = peer_schemas
    return mgr


def test_create_with_explicit_protocol_versions_skips_inference(tmp_path: Path):
    # No peers => inference would write only protocol 0; explicit versions override.
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)

    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        protocol_versions=["1"],
    )

    public_root = mgr.syftbox_config.datasite_public_root(DO_EMAIL) / "syft_datasets"
    # Exactly the requested version is written: v1 layout, not the flat default.
    assert (public_root / "v1" / "demo" / "dataset.yaml").exists()
    assert not (public_root / "demo").exists()
    assert mgr.get("demo")._ref.protocol_version == "1"


def test_migrate_dataset_v0_to_v1_preserves_identity(tmp_path: Path):
    # No peers => created in the widest-compatible (flat, protocol 0) layout.
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr.create(name="demo", mock_path=mock, private_path=private, readme_path=readme)

    ref0 = mgr.storage.find_dataset_ref(DO_EMAIL, "demo")
    assert ref0.protocol_version == "0"
    old = mgr.storage.read_dataset(ref0)

    migrated = mgr.storage.migrate_dataset(ref0, "1")

    public_root = mgr.syftbox_config.datasite_public_root(DO_EMAIL) / "syft_datasets"
    # Source (v0) copy is left intact; the v1 layout is created alongside it.
    assert (public_root / "demo" / "dataset.yaml").exists()
    assert (public_root / "v1" / "demo" / "dataset.yaml").exists()

    # Identity is preserved across the migration (datasets are immutable).
    assert migrated.uid == old.uid
    assert migrated.created_at == old.created_at
    assert migrated._ref.protocol_version == "1"

    # v1 dataset.yaml carries the identity fields (unlike flat protocol 0).
    raw = yaml.safe_load((public_root / "v1" / "demo" / "dataset.yaml").read_text())
    assert raw["canonical_name"] == "Dataset" and raw["version"] == "1"

    # Payload copied; metadata/permission/readme files excluded from mock_files.
    mock_names = sorted(p.name for p in migrated.mock_files)
    assert mock_names == ["mock.csv"]

    # Private data + a fresh private config land in the v1 layout.
    target_ref = mgr.storage.new_dataset_ref("demo", "1")
    assert mgr.storage.private_dataset_dir(target_ref).joinpath("private.csv").exists()
    assert mgr.storage.read_private_config(target_ref).uid == old.uid

    # get_all() dedupes the two on-disk copies, preferring the newest (v1).
    all_datasets = mgr.get_all()
    assert len(all_datasets) == 1
    assert all_datasets[0]._ref.protocol_version == "1"


def test_default_create_writes_protocol_0(tmp_path: Path):
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)

    dataset = mgr.create(
        name="demo", mock_path=mock, private_path=private, readme_path=readme
    )
    # Flat layout, no v<n>, no identity fields (byte-compatible with 0.1.20).
    assert dataset.mock_dir.parent.name == "syft_datasets"
    raw = yaml.safe_load((dataset.mock_dir / "dataset.yaml").read_text())
    assert "canonical_name" not in raw

    got = mgr.get("demo")
    assert got.name == "demo"
    assert got.version == dataset_registry.latest_version("Dataset")
    assert [p.name for p in got.mock_files] == ["mock.csv"]
    assert [p.name for p in got.private_files] == ["private.csv"]


def test_create_for_protocol1_peer_writes_v1(tmp_path: Path):
    schema1 = dataset_registry.schema_for_protocol_version("1")
    mgr = _dataset_manager(tmp_path, peer_schemas={DS1: schema1})
    mock, private, readme = _create_dataset_files(tmp_path)

    dataset = mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        users=[DS1],
    )
    # Versioned layout, identity fields present.
    assert dataset.mock_dir.parent.name == "v1"
    raw = yaml.safe_load((dataset.mock_dir / "dataset.yaml").read_text())
    assert raw["canonical_name"] == "Dataset" and raw["version"] == "1"

    # The shared peer has read access to the (v1) mock dir.
    perm = dataset.mock_dir / "syft.pub.yaml"
    assert perm.exists()
    assert DS1 in perm.read_text()

    got = mgr.get("demo")
    assert got.name == "demo" and got._ref.protocol_version == "1"


def test_multi_version_write_for_mixed_audience(tmp_path: Path):
    schema0 = dataset_registry.schema_for_protocol_version("0")
    schema1 = dataset_registry.schema_for_protocol_version("1")
    mgr = _dataset_manager(tmp_path, peer_schemas={DS0: schema0, DS1: schema1})
    mock, private, readme = _create_dataset_files(tmp_path)

    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        users=[DS0, DS1],
    )

    root = mgr.syftbox_config.datasite_public_root(DO_EMAIL) / "syft_datasets"
    # Both layouts written: flat (protocol 0) and v1 (protocol 1).
    assert (root / "demo" / "dataset.yaml").exists()
    assert (root / "v1" / "demo" / "dataset.yaml").exists()

    # get_all() dedupes the two on-disk copies to one, preferring the newest.
    all_datasets = mgr.get_all()
    assert len(all_datasets) == 1
    assert all_datasets[0]._ref.protocol_version == "1"
    assert mgr.get("demo")._ref.protocol_version == "1"


def test_delete_removes_all_protocol_versions(tmp_path: Path):
    schema0 = dataset_registry.schema_for_protocol_version("0")
    schema1 = dataset_registry.schema_for_protocol_version("1")
    mgr = _dataset_manager(tmp_path, peer_schemas={DS0: schema0, DS1: schema1})
    mock, private, readme = _create_dataset_files(tmp_path)

    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        users=[DS0, DS1],
    )

    public_root = mgr.syftbox_config.datasite_public_root(DO_EMAIL) / "syft_datasets"
    private_root = mgr.syftbox_config.datasite_private_root(DO_EMAIL) / "syft_datasets"
    # Both layouts exist on disk before deletion.
    assert (public_root / "demo").exists()
    assert (public_root / "v1" / "demo").exists()
    assert (private_root / "demo").exists()
    assert (private_root / "v1" / "demo").exists()

    mgr.delete(name="demo", require_confirmation=False)

    # Every protocol version is gone, public and private.
    assert not (public_root / "demo").exists()
    assert not (public_root / "v1" / "demo").exists()
    assert not (private_root / "demo").exists()
    assert not (private_root / "v1" / "demo").exists()
    assert mgr.get_all() == []
