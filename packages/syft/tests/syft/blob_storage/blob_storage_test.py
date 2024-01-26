# stdlib
import io

# third party
import pytest

# syft absolute
import syft as sy
from syft.service.context import AuthedServiceContext
from syft.service.response import SyftSuccess
from syft.service.user.user import UserCreate
from syft.store.blob_storage import BlobDeposit
from syft.store.blob_storage import SyftObjectRetrieval
from syft.types.blob_storage import CreateBlobStorageEntry

raw_data = {"test": "test"}
data = sy.serialize(raw_data, to_bytes=True)


@pytest.fixture
def authed_context(worker):
    return AuthedServiceContext(node=worker, credentials=worker.signing_key.verify_key)


@pytest.fixture
def blob_storage(worker):
    return worker.get_service("BlobStorageService")


def test_blob_storage_allocate(authed_context, blob_storage):
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    assert isinstance(blob_deposit, BlobDeposit)


def test_blob_storage_write(authed_context, blob_storage):
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    file_data = io.BytesIO(data)
    written_data = blob_deposit.write(file_data)

    assert isinstance(written_data, SyftSuccess)


def test_blob_storage_write_syft_object(authed_context, blob_storage):
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    user = UserCreate(email="info@openmined.org")
    file_data = io.BytesIO(sy.serialize(user, to_bytes=True))
    written_data = blob_deposit.write(file_data)

    assert isinstance(written_data, SyftSuccess)


def test_blob_storage_read(authed_context, blob_storage):
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    file_data = io.BytesIO(data)
    blob_deposit.write(file_data)

    syft_retrieved_data = blob_storage.read(
        authed_context, blob_deposit.blob_storage_entry_id
    )

    assert isinstance(syft_retrieved_data, SyftObjectRetrieval)
    assert syft_retrieved_data.read() == raw_data


# def test_blob_storage_delete(authed_context, blob_storage):
#     blob_data = CreateBlobStorageEntry.from_obj(data)
#     blob_deposit = blob_storage.allocate(authed_context, blob_data)
#     blob_storage.delete(authed_context, blob_deposit.blob_storage_entry_id)

#     with pytest.raises(FileNotFoundError):
#         blob_storage.read(authed_context, blob_deposit.blob_storage_entry_id)
