# stdlib
import io

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft import ActionObject
from syft import Dataset
from syft import Worker
from syft.client.datasite_client import DatasiteClient
from syft.service.blob_storage.util import can_upload_to_blob_storage
from syft.service.blob_storage.util import min_size_for_blob_storage_upload
from syft.service.context import AuthedServiceContext
from syft.service.response import SyftSuccess
from syft.service.user.user import UserCreate
from syft.store.blob_storage import BlobDeposit
from syft.store.blob_storage import SyftObjectRetrieval
from syft.types.blob_storage import CreateBlobStorageEntry
from syft.types.errors import SyftException

raw_data = {"test": "test"}
data = sy.serialize(raw_data, to_bytes=True)


@pytest.fixture
def authed_context(worker):
    yield AuthedServiceContext(server=worker, credentials=worker.signing_key.verify_key)


@pytest.fixture(scope="function")
def blob_storage(worker):
    yield worker.services.blob_storage


def test_blob_storage_allocate(authed_context, blob_storage):
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    assert isinstance(blob_deposit, BlobDeposit)


def test_blob_storage_write(worker):
    blob_storage = worker.services.blob_storage
    authed_context = AuthedServiceContext(
        server=worker, credentials=worker.signing_key.verify_key
    )
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    file_data = io.BytesIO(data)
    written_data = blob_deposit.write(file_data).unwrap()

    assert isinstance(written_data, SyftSuccess)

    worker.cleanup()


def test_blob_storage_write_syft_object(worker):
    blob_storage = worker.services.blob_storage
    authed_context = AuthedServiceContext(
        server=worker, credentials=worker.signing_key.verify_key
    )
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    user = UserCreate(email="info@openmined.org", name="Jana Doe", password="password")
    file_data = io.BytesIO(sy.serialize(user, to_bytes=True))
    written_data = blob_deposit.write(file_data).unwrap()

    assert isinstance(written_data, SyftSuccess)
    worker.cleanup()


def test_blob_storage_read(worker):
    blob_storage = worker.services.blob_storage
    authed_context = AuthedServiceContext(
        server=worker, credentials=worker.signing_key.verify_key
    )
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    file_data = io.BytesIO(data)
    blob_deposit.write(file_data).unwrap()

    syft_retrieved_data = blob_storage.read(
        authed_context, blob_deposit.blob_storage_entry_id
    )

    assert isinstance(syft_retrieved_data, SyftObjectRetrieval)
    assert syft_retrieved_data.read() == raw_data
    worker.cleanup()


def test_blob_storage_delete(authed_context, blob_storage):
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)

    assert isinstance(blob_deposit, BlobDeposit)

    file_data = io.BytesIO(data)
    written_data = blob_deposit.write(file_data).unwrap()
    assert type(written_data) is SyftSuccess

    item = blob_storage.read(authed_context, blob_deposit.blob_storage_entry_id)
    assert isinstance(item, SyftObjectRetrieval)
    assert item.read() == raw_data

    del_type = blob_storage.delete(authed_context, blob_deposit.blob_storage_entry_id)
    assert type(del_type) is SyftSuccess

    with pytest.raises(SyftException):
        blob_storage.read(authed_context, blob_deposit.blob_storage_entry_id)


def test_action_obj_send_save_to_blob_storage(worker):
    # this small object should not be saved to blob storage
    data_small: np.ndarray = np.array([1, 2, 3])
    action_obj = ActionObject.from_obj(data_small)
    assert action_obj.dtype == data_small.dtype
    root_client: DatasiteClient = worker.root_client
    action_obj.send(root_client)
    assert action_obj.syft_blob_storage_entry_id is None

    # big object that should be saved to blob storage (in mb)
    assert min_size_for_blob_storage_upload(root_client.api.metadata) == 1
    num_elements = 20 * 1024 * 1024
    data_big = np.random.randint(0, 100, size=num_elements)  # 4 bytes per int32
    action_obj_2 = ActionObject.from_obj(data_big)
    assert can_upload_to_blob_storage(action_obj_2, root_client.api.metadata).unwrap()
    action_obj_2.send(root_client)
    assert isinstance(action_obj_2.syft_blob_storage_entry_id, sy.UID)
    # get back the object from blob storage to check if it is the same
    root_authed_ctx = AuthedServiceContext(
        server=worker, credentials=root_client.verify_key
    )
    blob_storage = worker.services.blob_storage
    syft_retrieved_data = blob_storage.read(
        root_authed_ctx, action_obj_2.syft_blob_storage_entry_id
    )
    assert isinstance(syft_retrieved_data, SyftObjectRetrieval)
    assert all(syft_retrieved_data.read() == data_big)


def test_upload_dataset_save_to_blob_storage(
    worker: Worker, big_dataset: Dataset, small_dataset: Dataset
) -> None:
    root_client: DatasiteClient = worker.root_client
    # the small dataset should not be saved to the blob storage
    root_client.upload_dataset(small_dataset)
    assert len(root_client.api.services.blob_storage.get_all()) == 0

    # the big dataset should be saved to the blob storage
    root_client.upload_dataset(big_dataset)
    assert len(root_client.api.services.blob_storage.get_all()) == 2
