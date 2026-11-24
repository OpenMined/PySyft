import os
import random
from pathlib import Path

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient


def _create_tmp_dataset_files():
    tmp_dir = Path("/tmp/syft-immutability-testing") / str(random.randint(1, 1000000))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mock_path = tmp_dir / "mock.txt"
    private_path = tmp_dir / "private.txt"
    mock_path.write_text("Hello, world!")
    private_path.write_text("Hello, world private!")
    return mock_path, private_path


def test_enclave_blocks_reshare_of_private_dataset(private_dataset_dir):
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
    enclave._rds.sync()

    enclave_private_dir = private_dataset_dir(
        enclave._rds.syftbox_folder, do1.email, "testdataset"
    )
    assert enclave_private_dir is not None
    original_content = (enclave_private_dir / "private.txt").read_bytes()
    assert original_content == b"Hello, world private!"

    # DO modifies private data locally and re-shares
    private_path.write_text("TAMPERED DATA")
    do1._rds.dataset_manager.delete("testdataset", require_confirmation=False)
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
    enclave._rds.sync()

    # Enclave still has original content — overwrite was blocked
    assert (enclave_private_dir / "private.txt").read_bytes() == original_content
