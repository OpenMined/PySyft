"""End-to-end dataset flow through SyftDatasetManager.

Datasets are broadcast: one public copy is read by the whole audience. A create
therefore writes the current protocol, plus one older layout for each version
its audience reads. A peer that arrives after the create gets its layout from
the backfill, at the share.
"""

from pathlib import Path

import yaml
from syft_datasets.dataset_manager import SHARE_WITH_ANY, SyftDatasetManager
from syft_datasets.migrations import dataset_registry
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION

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
    # No peers => the default writes the current layout. The explicit list
    # overrides that, so an older layout alone is what lands.
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)

    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        protocol_versions=["0"],
    )

    public_root = mgr.syftbox_config.datasite_public_root(DO_EMAIL) / "syft_datasets"
    # Exactly the requested version is written: the flat layout, not the
    # current one the default would have chosen.
    assert (public_root / "demo" / "dataset.yaml").exists()
    assert not (public_root / f"v{DATASET_PROTOCOL_VERSION}").exists()
    assert mgr.get("demo")._ref.protocol_version == "0"


def test_migrate_dataset_v0_to_v1_preserves_identity(tmp_path: Path):
    # An explicit protocol 0 create, standing in for a dataset written by an
    # earlier release: the flat layout with no v<n> segment.
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        protocol_versions=["0"],
    )

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


def _v0_dataset(tmp_path: Path) -> SyftDatasetManager:
    """A manager holding one dataset in the flat protocol 0 layout."""
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        protocol_versions=["0"],
    )
    return mgr


def test_a_second_migrate_returns_the_existing_copy(tmp_path: Path):
    # A sweep can run twice, and a migrate must not raise the second time.
    mgr = _v0_dataset(tmp_path)
    ref0 = mgr.storage.find_dataset_ref(DO_EMAIL, "demo", protocol_version="0")

    first = mgr.storage.migrate_dataset(ref0, "1")
    second = mgr.storage.migrate_dataset(ref0, "1")

    assert second.uid == first.uid
    assert second.created_at == first.created_at
    assert second._ref.protocol_version == "1"
    # Still one copy for each layout, and the source survives.
    versions = sorted(
        r.protocol_version
        for r in mgr.storage.iter_dataset_refs_all_protocols(DO_EMAIL)
        if r.name == "demo"
    )
    assert versions == ["0", "1"]


def test_a_migrate_over_an_interrupted_copy_succeeds(tmp_path: Path):
    # The payload is copied before the metadata is written, so an interruption
    # leaves a target that no scan can see and that a plain copy refuses.
    mgr = _v0_dataset(tmp_path)
    ref0 = mgr.storage.find_dataset_ref(DO_EMAIL, "demo", protocol_version="0")
    target_ref = mgr.storage.new_dataset_ref("demo", "1")

    debris = mgr.storage.public_dataset_dir(target_ref)
    debris.mkdir(parents=True)
    (debris / "mock.csv").write_text("half a copy\n")
    assert not mgr.storage.metadata_path(target_ref).exists()

    migrated = mgr.storage.migrate_dataset(ref0, "1")

    assert migrated._ref.protocol_version == "1"
    assert mgr.storage.metadata_path(target_ref).exists()
    # The debris is replaced by a copy of the source, not merged with it.
    assert (debris / "mock.csv").read_text() == "id,value\n1,10\n"
    assert mgr.storage.read_private_config(target_ref).uid == migrated.uid


def test_a_migrate_over_a_target_with_no_private_config_redoes_the_copy(
    tmp_path: Path,
):
    # The public metadata is written before the private config, so a target with
    # only the first one is also incomplete.
    mgr = _v0_dataset(tmp_path)
    ref0 = mgr.storage.find_dataset_ref(DO_EMAIL, "demo", protocol_version="0")
    target_ref = mgr.storage.new_dataset_ref("demo", "1")
    mgr.storage.migrate_dataset(ref0, "1")
    mgr.storage.private_metadata_path(target_ref).unlink()

    mgr.storage.migrate_dataset(ref0, "1")

    assert mgr.storage.read_private_config(target_ref).uid == (
        mgr.storage.read_dataset(target_ref).uid
    )


def test_default_create_writes_the_current_protocol(tmp_path: Path):
    # No audience, so no older layout is needed. A peer that arrives later gets
    # its layout from the backfill, at the share.
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)

    dataset = mgr.create(
        name="demo", mock_path=mock, private_path=private, readme_path=readme
    )
    # Versioned layout, with the identity fields protocol 1 carries.
    assert dataset.mock_dir.parent.name == f"v{DATASET_PROTOCOL_VERSION}"
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


def test_the_audience_reads_back_from_the_ruleset(tmp_path: Path):
    # upgrade() must know who a dataset was shared with, and the ruleset of the
    # source layout records the explicit grants.
    schema0 = dataset_registry.schema_for_protocol_version("0")
    mgr = _dataset_manager(tmp_path, peer_schemas={DS0: schema0})
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        users=[DS0],
    )

    assert mgr.recover_audience_from_ruleset("demo") == [DS0]
    assert mgr.recover_audience_from_ruleset("demo", protocol_version="0") == [DS0]


def test_an_owner_only_dataset_recovers_an_empty_audience(tmp_path: Path):
    # No ruleset is written when there is no audience, and that is a real
    # answer: the dataset must stay owner-only.
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr.create(name="demo", mock_path=mock, private_path=private, readme_path=readme)

    assert mgr.recover_audience_from_ruleset("demo") == []


def test_a_dataset_shared_with_any_recovers_the_any_marker(tmp_path: Path):
    mgr = _dataset_manager(tmp_path)
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        users=SHARE_WITH_ANY,
    )

    assert mgr.recover_audience_from_ruleset("demo") == SHARE_WITH_ANY


def test_a_grant_reaches_the_ruleset_of_every_layout(tmp_path: Path):
    # The drift fix: a share after create must record the new user on every
    # layout, or upgrade() recovers the create-time audience and the promoted
    # copy never reaches that user.
    schema0 = dataset_registry.schema_for_protocol_version("0")
    mgr = _dataset_manager(tmp_path, peer_schemas={DS0: schema0})
    mock, private, readme = _create_dataset_files(tmp_path)
    mgr.create(
        name="demo",
        mock_path=mock,
        private_path=private,
        readme_path=readme,
        users=[DS0],
    )

    mgr.grant_read_on_every_layout("demo", ["late@test.org"])

    for protocol_version in ("0", "1"):
        assert sorted(
            mgr.recover_audience_from_ruleset("demo", protocol_version=protocol_version)
        ) == sorted([DS0, "late@test.org"])


def test_a_migrate_to_the_same_layout_keeps_the_source(tmp_path: Path):
    # The debris check clears an incomplete target, so a migrate onto the
    # source's own layout must never reach it.
    mgr = _v0_dataset(tmp_path)
    ref0 = mgr.storage.find_dataset_ref(DO_EMAIL, "demo", protocol_version="0")
    mgr.storage.private_metadata_path(ref0).unlink()

    same = mgr.storage.migrate_dataset(ref0, "0")

    assert same.name == "demo"
    assert mgr.storage.metadata_path(ref0).exists()
    assert (mgr.storage.public_dataset_dir(ref0) / "mock.csv").exists()
    assert (mgr.storage.private_dataset_dir(ref0) / "private.csv").exists()
