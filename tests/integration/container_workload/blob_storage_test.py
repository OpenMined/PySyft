# stdlib
import os

# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.skipif(
    "AZURE_BLOB_STORAGE_KEY" not in os.environ
    or os.environ["AZURE_BLOB_STORAGE_KEY"] == "",
    reason="AZURE_BLOB_STORAGE_KEY is not set",
)
@pytest.mark.container_workload
def test_mount_azure_blob_storage(datasite_1_port):
    datasite_client = sy.login(
        email="info@openmined.org", password="changethis", port=datasite_1_port
    )

    azure_storage_key = os.environ.get("AZURE_BLOB_STORAGE_KEY", None)
    assert azure_storage_key

    datasite_client.api.services.blob_storage.mount_azure(
        account_name="citestingstorageaccount",
        container_name="citestingcontainer",
        account_key=azure_storage_key,
        bucket_name="helmazurebucket",
    )
    blob_files = datasite_client.api.services.blob_storage.get_files_from_bucket(
        bucket_name="helmazurebucket"
    )
    assert isinstance(blob_files, list), blob_files
    assert len(blob_files) > 0
    document = [f for f in blob_files if "testfile.txt" in f.file_name][0]
    assert document.read() == b"abc\n"
