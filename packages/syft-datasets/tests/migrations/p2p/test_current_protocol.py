"""End-to-end dataset flow through SyftDatasetManager.

Datasets are broadcast: one public copy is read by the whole audience. A create
therefore writes the current protocol, plus one older layout for each version
its audience reads. A peer that arrives after the create gets its layout from
the backfill, at the share.
"""

from pathlib import Path

import yaml
from syft_datasets.config import protocol_dir_name
from syft_datasets.dataset_manager import SyftDatasetManager
from syft_datasets.migrations import dataset_registry
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION

DO_EMAIL = "do@test.org"
DS = "new@test.org"
SEGMENT = protocol_dir_name(DATASET_PROTOCOL_VERSION)


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
        mgr.storage.peer_schema = peer_schemas.get
    return mgr


def test_default_create_writes_the_current_protocol(tmp_path: Path):
    # No audience, so no older layout is needed. A peer that arrives later gets
    # its layout from the backfill, at the share.
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)

    dataset = mgr.create(
        name="demo", mock_path=mock, private_path=private, readme_path=readme
    )
    # Versioned layout, with the identity fields.
    assert dataset.mock_dir.parent.name == SEGMENT
    raw = yaml.safe_load((dataset.mock_dir / "dataset.yaml").read_text())
    assert raw["canonical_name"] == "Dataset"

    # The current layout is the only one written.
    versions = sorted(
        r.protocol_version
        for r in mgr.storage.iter_dataset_refs_all_protocols(DO_EMAIL)
        if r.name == "demo"
    )
    assert versions == [DATASET_PROTOCOL_VERSION]

    got = mgr.get("demo")
    assert got.name == "demo"
    assert got.version == dataset_registry.latest_version("Dataset")
    assert [p.name for p in got.mock_files] == ["mock.csv"]
    assert [p.name for p in got.private_files] == ["private.csv"]


def test_create_for_a_current_peer_writes_the_current_layout(tmp_path: Path):
    current_schema = dataset_registry.compute_protocol_schema()
    mgr = _dataset_manager(tmp_path, peer_schemas={DS: current_schema})
    mock, private, readme = _create_dataset_files(tmp_path)

    dataset = mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        users=[DS],
    )
    # Versioned layout, identity fields present.
    assert dataset.mock_dir.parent.name == SEGMENT
    raw = yaml.safe_load((dataset.mock_dir / "dataset.yaml").read_text())
    assert raw["canonical_name"] == "Dataset"
    assert raw["version"] == dataset_registry.latest_version("Dataset")

    # The shared peer has read access to the mock dir.
    perm = dataset.mock_dir / "syft.pub.yaml"
    assert perm.exists()
    assert DS in perm.read_text()

    got = mgr.get("demo")
    assert got.name == "demo" and got._ref.protocol_version == DATASET_PROTOCOL_VERSION


def test_private_files_wire_private_metadata_in_protocol_format(tmp_path: Path):
    """get_private_dataset_files serializes private_metadata through the codec."""
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr = _dataset_manager(tmp_path)
    mgr.create(name="demo", mock_path=mock, private_path=private, readme_path=readme)

    files = mgr.get_private_dataset_files(
        "demo", protocol_version=DATASET_PROTOCOL_VERSION
    )
    metas = [
        content
        for path, content in files.items()
        if path.name == "private_metadata.yaml"
    ]
    assert len(metas) == 1
    raw = yaml.safe_load(metas[0])
    assert raw["canonical_name"] == "PrivateDatasetConfig"
    assert raw["version"] == dataset_registry.latest_version("PrivateDatasetConfig")
    assert not Path(str(raw["data_dir"])).is_absolute()


def test_delete_removes_the_dataset(tmp_path: Path):
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr.create(name="demo", mock_path=mock, private_path=private, readme_path=readme)

    public_root = mgr.syftbox_config.datasite_public_root(DO_EMAIL) / "syft_datasets"
    private_root = mgr.syftbox_config.datasite_private_root(DO_EMAIL) / "syft_datasets"
    assert (public_root / SEGMENT / "demo").exists()
    assert (private_root / SEGMENT / "demo").exists()

    mgr.delete(name="demo", require_confirmation=False)

    assert not (public_root / SEGMENT / "demo").exists()
    assert not (private_root / SEGMENT / "demo").exists()
    assert mgr.get_all() == []
