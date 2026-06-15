"""Tests for hidden/generated file exclusion in sync and notifications."""

import tempfile
from pathlib import Path

from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.utils.path_filters import (
    is_excluded_path,
    is_normal_syncable_path,
)


def test_is_normal_syncable_path_excludes_hidden():
    assert is_normal_syncable_path("ok/main.py") is True
    assert is_normal_syncable_path(".venv/lib/site.py") is False
    assert is_normal_syncable_path("a/__pycache__/x.pyc") is False


def test_is_normal_syncable_path_excludes_private():
    assert is_normal_syncable_path("private/secret.txt") is False
    assert is_normal_syncable_path("public/private.txt") is True
    # "private" as a non-first component is still syncable
    assert is_normal_syncable_path("apis/private/x.py") is True


def test_is_normal_syncable_path_excludes_collections():
    coll = Path("public/syft_datasets")
    assert is_normal_syncable_path("public/syft_datasets/foo/bar.csv", coll) is False
    assert is_normal_syncable_path("public/syft_datasets", coll) is False
    # Sibling with a similar name must NOT match
    assert is_normal_syncable_path("public/syft_datasets_backup/x", coll) is True
    # None disables the collections check
    assert is_normal_syncable_path("public/syft_datasets/foo/bar.csv", None) is True


def _create_job_dir_with_hidden_files(base: Path) -> Path:
    """Create a fake job directory containing both normal and hidden files."""
    job_dir = base / "my_job"
    job_dir.mkdir(parents=True)

    # Normal files that should be included
    (job_dir / "main.py").write_text("print('hello')")
    (job_dir / "utils.py").write_text("x = 1")

    # Hidden / generated dirs that should be excluded
    venv = job_dir / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "site.py").write_text("# venv")

    git = job_dir / ".git"
    git.mkdir()
    (git / "config").write_text("[core]")

    pycache = job_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-312.pyc").write_bytes(b"\x00")

    return job_dir


def test_read_job_code_skips_hidden_files(tmp_path):
    """_read_job_code should not include files under excluded directories."""
    from syft_bg.notify.handlers.job import _read_job_code
    from unittest.mock import MagicMock

    # Build a syftbox-like layout: <syftbox>/<email>/<job_dir>
    email = "alice@example.com"
    syftbox = tmp_path / "syftbox"
    datasite = syftbox / email
    datasite.mkdir(parents=True)
    job_dir = _create_job_dir_with_hidden_files(datasite)

    mock_job = MagicMock()
    mock_job.name = "my_job"
    mock_job.code_dir = job_dir

    mock_client = MagicMock()
    mock_client.jobs = [mock_job]
    mock_client.config.syftbox_folder = syftbox
    mock_client.config.current_user_email = email

    result = _read_job_code(mock_client, "my_job")

    assert result is not None
    assert "main.py" in result
    assert "utils.py" in result
    assert len(result) == 2, f"expected only 2 files, got {sorted(result)}"

    # None of the hidden/generated files should appear
    for key in result:
        assert not is_excluded_path(key), f"excluded file leaked through: {key}"


def test_push_job_files_pushes_all_files_and_warns(caplog):
    """push_job_files should push every file but log a warning for non-syncable ones."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()

    with tempfile.TemporaryDirectory() as tmp:
        job_dir = _create_job_dir_with_hidden_files(Path(tmp))

        # Place the job dir inside ds_manager's syftbox folder so relative paths work
        target = (
            Path(ds_manager.syftbox_folder)
            / do_manager.email
            / "app_data"
            / "job"
            / "inbox"
            / ds_manager.email
            / "my.job"
        )
        import shutil

        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(job_dir, target)

        with caplog.at_level("WARNING", logger="syft_client.sync.syftbox_manager"):
            ds_manager.push_job_files(target)
        do_manager.sync()

        # Normal files still arrive on the DO side.
        event_messages = do_manager._get_all_accepted_events_do()
        arrived_paths = [
            str(e.path_in_datasite) for msg in event_messages for e in msg.events
        ]
        assert any("main.py" in p for p in arrived_paths)
        assert any("utils.py" in p for p in arrived_paths)

        # Non-syncable paths must NOT make it through to the DO side.
        for p in arrived_paths:
            assert ".venv" not in p, f"non-syncable file arrived: {p}"
            assert ".git" not in p, f"non-syncable file arrived: {p}"
            assert "__pycache__" not in p, f"non-syncable file arrived: {p}"

        # A warning is emitted for each non-syncable file (we no longer filter).
        warning_messages = [
            r.getMessage() for r in caplog.records if r.levelname == "WARNING"
        ]
        non_syncable_warnings = [
            m for m in warning_messages if "non-syncable path" in m
        ]
        assert any(".venv" in m for m in non_syncable_warnings)
        assert any(".git" in m for m in non_syncable_warnings)
        assert any("__pycache__" in m for m in non_syncable_warnings)


def test_owner_cache_get_syncable_paths_filters_all_three_categories(tmp_path):
    """DataSiteOwnerEventCache.get_syncable_paths excludes excluded/private/collections."""
    from syft_client.sync.sync.caches.datasite_owner_cache import (
        DataSiteOwnerEventCache,
    )
    from syft_client.sync.sync.caches.cache_file_writer_connection import (
        InMemoryCacheFileConnection,
    )

    email = "alice@example.com"
    syftbox = tmp_path / "syftbox"
    collections_folder = syftbox / email / "public" / "syft_datasets"

    file_conn = InMemoryCacheFileConnection()
    file_conn.write_file("ok/main.py", b"hi")
    file_conn.write_file(".venv/lib/x.py", b"x")
    file_conn.write_file("private/secret.txt", b"s")
    file_conn.write_file("public/syft_datasets/d1/data.csv", b"d")
    file_conn.write_file("public/other/keep.txt", b"k")

    cache = DataSiteOwnerEventCache(
        file_connection=file_conn,
        email=email,
        syftbox_folder=syftbox,
        collections_folder=collections_folder,
    )

    syncable = {str(p) for p in cache.get_syncable_paths()}
    assert syncable == {"ok/main.py", "public/other/keep.txt"}


def test_create_checkpoint_excludes_collections(tmp_path):
    """create_checkpoint must drop files under the collections folder."""
    from syft_client.sync.sync.caches.datasite_owner_cache import (
        DataSiteOwnerEventCache,
    )
    from syft_client.sync.sync.caches.cache_file_writer_connection import (
        InMemoryCacheFileConnection,
    )

    email = "alice@example.com"
    syftbox = tmp_path / "syftbox"
    collections_folder = syftbox / email / "public" / "syft_datasets"

    file_conn = InMemoryCacheFileConnection()
    file_conn.write_file("ok/main.py", b"hi")
    file_conn.write_file("public/syft_datasets/d1/data.csv", b"d")

    cache = DataSiteOwnerEventCache(
        file_connection=file_conn,
        email=email,
        syftbox_folder=syftbox,
        collections_folder=collections_folder,
    )
    # Populate file_hashes so the checkpoint considers these paths.
    cache.file_hashes["ok/main.py"] = "h1"
    cache.file_hashes["public/syft_datasets/d1/data.csv"] = "h2"

    checkpoint = cache.create_checkpoint()
    checkpoint_paths = {f.path for f in checkpoint.files}
    assert "ok/main.py" in checkpoint_paths
    assert not any("syft_datasets" in p for p in checkpoint_paths), (
        f"checkpoint should not contain collections paths, got: {checkpoint_paths}"
    )
