import os
import random
from pathlib import Path

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient
from syft_enclaves.immutability import (
    is_private_dataset_path,
    make_private_dataset_immutability_filter,
)


# --- Unit tests for is_private_dataset_path ---


def test_is_private_dataset_path_positive():
    assert is_private_dataset_path(
        "do@example.com/private/syft_datasets/my_ds/data.csv"
    )


def test_is_private_dataset_path_public():
    assert not is_private_dataset_path(
        "do@example.com/public/syft_datasets/my_ds/data.csv"
    )


def test_is_private_dataset_path_wrong_subdir():
    assert not is_private_dataset_path("do@example.com/private/other/file.txt")


def test_is_private_dataset_path_too_short():
    # Missing the file component — just the dataset dir itself
    assert not is_private_dataset_path("do@example.com/private/syft_datasets/my_ds")


# --- Unit tests for the filter ---


def test_filter_allows_first_write(tmp_path):
    f = make_private_dataset_immutability_filter(tmp_path)
    path = "do@example.com/private/syft_datasets/my_ds/data.csv"
    assert f(path, False) is True


def test_filter_blocks_overwrite(tmp_path):
    f = make_private_dataset_immutability_filter(tmp_path)
    path = "do@example.com/private/syft_datasets/my_ds/data.csv"

    full_path = tmp_path / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text("original data")

    assert f(path, False) is False


def test_filter_blocks_delete(tmp_path):
    f = make_private_dataset_immutability_filter(tmp_path)
    path = "do@example.com/private/syft_datasets/my_ds/data.csv"

    full_path = tmp_path / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text("original data")

    assert f(path, True) is False


def test_filter_allows_non_dataset_paths(tmp_path):
    f = make_private_dataset_immutability_filter(tmp_path)

    assert f("do@example.com/public/syft_datasets/ds/file.csv", False) is True
    assert f("do@example.com/app_data/job/inbox/ds/job1/run.sh", False) is True


def test_filter_allows_remaining_files_in_batch(tmp_path):
    """First file creates the dir; other files in the same dataset should still be allowed."""
    f = make_private_dataset_immutability_filter(tmp_path)
    base = "do@example.com/private/syft_datasets/my_ds"

    # Simulate first file already written
    first = tmp_path / base / "data.csv"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("data")

    # Second file doesn't exist yet — allowed
    assert f(f"{base}/metadata.yaml", False) is True

    # First file exists — blocked
    assert f(f"{base}/data.csv", False) is False


# --- Integration test ---


def _create_tmp_dataset_files():
    tmp_dir = Path("/tmp/syft-immutability-testing") / str(random.randint(1, 1000000))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mock_path = tmp_dir / "mock.txt"
    private_path = tmp_dir / "private.txt"
    mock_path.write_text("Hello, world!")
    private_path.write_text("Hello, world private!")
    return mock_path, private_path


def test_enclave_blocks_reshare_of_private_dataset():
    """After DO shares private data with enclave, a second share should not overwrite."""
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    mock_path, private_path = _create_tmp_dataset_files()

    do1.create_dataset(
        name="testdataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="Test dataset",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )

    # First share + sync — enclave receives private data
    do1.share_private_dataset("testdataset", enclave.email)
    enclave._manager.sync()

    enclave_private_dir = (
        enclave._manager.syftbox_folder
        / do1.email
        / "private"
        / "syft_datasets"
        / "testdataset"
    )
    assert enclave_private_dir.exists()
    original_content = (enclave_private_dir / "private.txt").read_bytes()
    assert original_content == b"Hello, world private!"

    # DO modifies private data locally and re-shares
    private_path.write_text("TAMPERED DATA")
    do1._manager.dataset_manager.delete("testdataset", require_confirmation=False)
    do1.create_dataset(
        name="testdataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="Test dataset",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )
    do1.share_private_dataset("testdataset", enclave.email)
    enclave._manager.sync()

    # Enclave still has original content — overwrite was blocked
    assert (enclave_private_dir / "private.txt").read_bytes() == original_content
