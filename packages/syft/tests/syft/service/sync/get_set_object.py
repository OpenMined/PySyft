# third party
import numpy as np
import pytest

# syft absolute
import syft
import syft as sy
from syft.client.datasite_client import DatasiteClient
from syft.client.sync_decision import SyncDecision
from syft.client.syncing import compare_clients
from syft.client.syncing import resolve
from syft.server.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.code.user_code import ApprovalDecision
from syft.service.code.user_code import UserCodeStatus
from syft.service.dataset.dataset import Dataset
from syft.service.job.job_stash import Job
from syft.service.request.request import RequestStatus
from syft.service.response import SyftSuccess
from syft.service.sync.resolve_widget import ResolveWidget
from syft.service.user.user import User, UserView
from syft.types.errors import SyftException


def get_ds_client(client: DatasiteClient) -> DatasiteClient:
    client.register(
        name="a",
        email="a@a.com",
        password="asdf",
        password_verify="asdf",
    )
    return client.login(email="a@a.com", password="asdf")


def test_get_set_object(high_worker):
    high_client: DatasiteClient = high_worker.root_client
    _ = get_ds_client(high_client)
    root_datasite_client = high_worker.root_client
    dataset = sy.Dataset(
        name="local_test",
        asset_list=[
            sy.Asset(
                name="local_test",
                data=[1, 2, 3],
                mock=[1, 1, 1],
            )
        ],
    )
    root_datasite_client.upload_dataset(dataset)
    dataset = root_datasite_client.datasets[0]
    
    other_dataset = high_client.api.services.sync._get_object(uid=dataset.id, object_type=Dataset)
    other_dataset.server_uid = dataset.server_uid
    assert dataset == other_dataset
    other_dataset.name = "new_name"
    updated_dataset = high_client.api.services.sync._update_object(
        object=other_dataset
    )
    assert updated_dataset.name == "new_name"

    asset = root_datasite_client.datasets[0].assets[0]
    source_ao = high_client.api.services.action.get(uid=asset.action_id)
    ao = high_client.api.services.sync._get_object(
        uid=asset.action_id, object_type=ActionObject
    )
    ao._set_obj_location_(
            high_worker.id,
            root_datasite_client.credentials,
        )
    assert source_ao == ao
