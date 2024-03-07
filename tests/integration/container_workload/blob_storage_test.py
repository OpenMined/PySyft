import pytest
import os


import syft as sy

def test_mount_azure_blob_storage(domain_1_port):
    domain_client = sy.login(email="info@openmined.org", password="changethis", port=domain_1_port)
    domain_client.api.services.blob_storage.mount_azure(
        account_name="citestingstorageaccount",
        container_name="citestingcontainer",
        account_key=os.environ["AZURE_BLOB_STORAGE_KEY"],
        bucket_name="helmazurebucket",
    )
    blob_files = domain_client.api.services.blob_storage.get_files_from_bucket(
        bucket_name="helmazurebucket"
    )
    document = [f for f in blob_files if "testfile.txt" in f.file_name][0]
    assert document.read() == b"abc\n"