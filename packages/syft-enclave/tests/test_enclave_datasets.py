import os
import random
from pathlib import Path

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient


def create_tmp_dataset_files():
    tmp_dir = Path("/tmp/syft-datasets-testing") / str(random.randint(1, 1000000))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mock_path = tmp_dir / "mock.txt"
    private_path = tmp_dir / "private.txt"
    mock_path.write_text("Hello, world!")
    private_path.write_text("Hello, world private!")
    return mock_path, private_path


def test_share_private_dataset_with_enclave():
    """Test full flow: DO creates dataset, shares private data with enclave, enclave can access it."""
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    mock_path, private_path = create_tmp_dataset_files()

    # DO1 creates dataset with mock + private data (sync=False to avoid issue with
    # the enclave DO inbox not being visible yet during the sync inside create_dataset)
    do1.create_dataset(
        name="testdataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="Test dataset",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )

    # DO1 + DS sync -> DS can see mock data
    do1.sync()
    ds._manager.sync()
    ds_datasets = ds.datasets.get_all()
    assert len(ds_datasets) == 1
    assert ds_datasets[0].name == "testdataset"
    assert len(ds_datasets[0].mock_files) == 1

    # DS can read mock file content
    ds_dataset = ds.datasets.get("testdataset", datasite=do1.email)
    assert ds_dataset is not None
    mock_content = ds_dataset.mock_files[0].read_text()
    assert mock_content == "Hello, world!"

    non_existing_ds_private_dir = (
        ds._manager.syftbox_folder
        / do1.email
        / "private"
        / "syft_datasets"
        / "testdataset"
    )
    assert not non_existing_ds_private_dir.exists()

    # DO1 shares private dataset with enclave
    do1.share_private_dataset("testdataset", enclave.email)

    # Enclave syncs (as DS) and pulls the private data events from DO1's outbox
    enclave._manager.sync()

    # Enclave can see the dataset via mock data (shared with DS and enclave shares peers)
    # But more importantly, enclave can access private files via shared_private_dir
    enclave_private_dir = (
        enclave._manager.syftbox_folder
        / do1.email
        / "private"
        / "syft_datasets"
        / "testdataset"
    )
    assert enclave_private_dir.exists()
    private_files = list(enclave_private_dir.iterdir())
    file_names = {f.name for f in private_files}
    assert "private.txt" in file_names
    assert "syft.pub.yaml" in file_names

    # Verify data content matches
    assert (
        enclave_private_dir / "private.txt"
    ).read_bytes() == b"Hello, world private!"
