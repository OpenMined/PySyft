# stdlib
import io
import os
import random

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft import ActionObject
from syft.client.domain_client import DomainClient
from syft.service.context import AuthedServiceContext
from syft.service.response import SyftSuccess
from syft.service.user.user import UserCreate
from syft.store.blob_storage import BlobDeposit
from syft.store.blob_storage import SyftObjectRetrieval
from syft.types.blob_storage import CreateBlobStorageEntry
from syft.util.util import min_size_for_blob_storage_upload

raw_data = {"test": "test"}
data = sy.serialize(raw_data, to_bytes=True)


@pytest.fixture
def authed_context(worker):
    yield AuthedServiceContext(node=worker, credentials=worker.signing_key.verify_key)


@pytest.fixture(scope="function")
def blob_storage(worker):
    yield worker.get_service("BlobStorageService")


def test_blob_storage_allocate(authed_context, blob_storage):
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    assert isinstance(blob_deposit, BlobDeposit)


def test_blob_storage_write():
    random.seed()
    name = "".join(str(random.randint(0, 9)) for i in range(8))
    worker = sy.Worker.named(name=name)
    blob_storage = worker.get_service("BlobStorageService")
    authed_context = AuthedServiceContext(
        node=worker, credentials=worker.signing_key.verify_key
    )
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    file_data = io.BytesIO(data)
    written_data = blob_deposit.write(file_data)

    assert isinstance(written_data, SyftSuccess)

    worker.cleanup()


def test_blob_storage_write_syft_object():
    random.seed()
    name = "".join(str(random.randint(0, 9)) for i in range(8))
    worker = sy.Worker.named(name=name)
    blob_storage = worker.get_service("BlobStorageService")
    authed_context = AuthedServiceContext(
        node=worker, credentials=worker.signing_key.verify_key
    )
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    user = UserCreate(email="info@openmined.org", name="Jana Doe", password="password")
    file_data = io.BytesIO(sy.serialize(user, to_bytes=True))
    written_data = blob_deposit.write(file_data)

    assert isinstance(written_data, SyftSuccess)
    worker.cleanup()


def test_blob_storage_read():
    random.seed()
    name = "".join(str(random.randint(0, 9)) for i in range(8))
    worker = sy.Worker.named(name=name)
    blob_storage = worker.get_service("BlobStorageService")
    authed_context = AuthedServiceContext(
        node=worker, credentials=worker.signing_key.verify_key
    )
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    file_data = io.BytesIO(data)
    blob_deposit.write(file_data)

    syft_retrieved_data = blob_storage.read(
        authed_context, blob_deposit.blob_storage_entry_id
    )

    assert isinstance(syft_retrieved_data, SyftObjectRetrieval)
    assert syft_retrieved_data.read() == raw_data
    worker.cleanup()


def test_blob_storage_delete(authed_context, blob_storage):
    blob_data = CreateBlobStorageEntry.from_obj(data)
    blob_deposit = blob_storage.allocate(authed_context, blob_data)
    blob_storage.delete(authed_context, blob_deposit.blob_storage_entry_id)

    with pytest.raises(FileNotFoundError):
        blob_storage.read(authed_context, blob_deposit.blob_storage_entry_id)


def test_action_obj_send_save_to_blob_storage(worker):
    # set this so we will always save action objects to blob storage
    os.environ["MIN_SIZE_BLOB_STORAGE_MB"] = "0"

    orig_obj: np.ndarray = np.array([1, 2, 3])
    action_obj = ActionObject.from_obj(orig_obj)
    assert action_obj.dtype == orig_obj.dtype

    root_client: DomainClient = worker.root_client
    action_obj.send(root_client)
    assert isinstance(action_obj.syft_blob_storage_entry_id, sy.UID)
    root_authed_ctx = AuthedServiceContext(
        node=worker, credentials=root_client.verify_key
    )

    blob_storage = worker.get_service("BlobStorageService")
    syft_retrieved_data = blob_storage.read(
        root_authed_ctx, action_obj.syft_blob_storage_entry_id
    )
    assert isinstance(syft_retrieved_data, SyftObjectRetrieval)
    assert all(syft_retrieved_data.read() == orig_obj)

    # stop saving small action objects to blob storage
    del os.environ["MIN_SIZE_BLOB_STORAGE_MB"]
    assert min_size_for_blob_storage_upload() == 16
    orig_obj_2: np.ndarray = np.array([1, 2, 4])
    action_obj_2 = ActionObject.from_obj(orig_obj_2)
    action_obj_2.send(root_client)
    assert action_obj_2.syft_blob_storage_entry_id is None
