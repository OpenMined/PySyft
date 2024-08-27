# stdlib
import random
from typing import Any
from uuid import uuid4

# third party
import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
import torch

# syft absolute
import syft as sy
from syft.server.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.action.action_object import TwinMode
from syft.service.blob_storage.util import can_upload_to_blob_storage
from syft.service.dataset.dataset import CreateAsset as Asset
from syft.service.dataset.dataset import CreateDataset as Dataset
from syft.service.dataset.dataset import _ASSET_WITH_NONE_MOCK_ERROR_MESSAGE
from syft.service.response import SyftSuccess
from syft.types.errors import SyftException


def random_hash() -> str:
    return uuid4().hex


def data():
    return np.array([1, 2, 3])


def mock():
    return np.array([1, 1, 1])


def make_asset_without_mock() -> dict[str, Any]:
    return {
        "name": random_hash(),
        "data": data(),
    }


def make_asset_with_mock() -> dict[str, Any]:
    return {**make_asset_without_mock(), "mock": mock()}


def make_asset_with_empty_mock() -> dict[str, Any]:
    return {**make_asset_without_mock(), "mock": ActionObject.empty()}


asset_without_mock = pytest.fixture(make_asset_without_mock)
asset_with_mock = pytest.fixture(make_asset_with_mock)
asset_with_empty_mock = pytest.fixture(make_asset_with_empty_mock)


@pytest.mark.parametrize(
    "asset_without_mock",
    [
        make_asset_without_mock(),
        {**make_asset_without_mock(), "mock": ActionObject.empty()},
    ],
)
def test_asset_without_mock_mock_is_real_must_be_false(
    asset_without_mock: dict[str, Any],
):
    asset = Asset(**asset_without_mock, mock_is_real=True)
    asset.mock_is_real = True
    assert not asset.mock_is_real


def test_mock_always_not_real_after_calling_no_mock(
    asset_with_mock: dict[str, Any],
) -> None:
    asset = Asset(**asset_with_mock, mock_is_real=True)
    assert asset.mock_is_real

    asset.no_mock()
    assert not asset.mock_is_real


def test_mock_always_not_real_after_set_mock_to_empty(
    asset_with_mock: dict[str, Any],
) -> None:
    asset = Asset(**asset_with_mock, mock_is_real=True)
    assert asset.mock_is_real

    asset.no_mock()
    assert not asset.mock_is_real

    asset.mock_is_real = True
    assert not asset.mock_is_real

    asset.mock = mock()
    asset.mock_is_real = True
    assert asset.mock_is_real


def test_mock_always_not_real_after_set_to_empty(
    asset_with_mock: dict[str, Any],
) -> None:
    asset = Asset(**asset_with_mock, mock_is_real=True)
    assert asset.mock_is_real

    asset.mock = ActionObject.empty()
    assert not asset.mock_is_real

    asset.mock_is_real = True
    assert not asset.mock_is_real

    asset.mock = mock()
    asset.mock_is_real = True
    assert asset.mock_is_real


@pytest.mark.parametrize(
    "empty_mock",
    [
        None,
        ActionObject.empty(),
    ],
)
def test_cannot_set_empty_mock_with_true_mock_is_real(
    asset_with_mock: dict[str, Any], empty_mock: Any
) -> None:
    asset = Asset(**asset_with_mock, mock_is_real=True)
    assert asset.mock_is_real

    with pytest.raises(SyftException) as exc:
        asset.set_mock(empty_mock, mock_is_real=True)

    assert asset.mock is asset_with_mock["mock"]
    assert exc.type == SyftException
    assert exc.value.public_message


def test_dataset_cannot_have_assets_with_none_mock() -> None:
    TOTAL_ASSETS = 10
    ASSETS_WITHOUT_MOCK = random.randint(2, 8)
    ASSETS_WITH_MOCK = TOTAL_ASSETS - ASSETS_WITHOUT_MOCK

    assets_without_mock = [
        Asset(**make_asset_without_mock()) for _ in range(ASSETS_WITHOUT_MOCK)
    ]
    assets_with_mock = [
        Asset(**make_asset_with_mock()) for _ in range(ASSETS_WITH_MOCK)
    ]
    assets = assets_without_mock + assets_with_mock

    with pytest.raises(ValidationError) as excinfo:
        Dataset(
            name=random_hash(),
            asset_list=assets,
        )

    assert _ASSET_WITH_NONE_MOCK_ERROR_MESSAGE in str(excinfo.value)

    assert Dataset(name=random_hash(), asset_list=assets_with_mock)


def test_dataset_can_have_assets_with_empty_mock() -> None:
    TOTAL_ASSETS = 10
    ASSETS_WITH_EMPTY_MOCK = random.randint(2, 8)
    ASSETS_WITH_MOCK = TOTAL_ASSETS - ASSETS_WITH_EMPTY_MOCK

    assets_without_mock = [
        Asset(**make_asset_without_mock(), mock=ActionObject.empty())
        for _ in range(ASSETS_WITH_EMPTY_MOCK)
    ]
    assets_with_mock = [
        Asset(**make_asset_with_mock()) for _ in range(ASSETS_WITH_MOCK)
    ]
    assets = assets_without_mock + assets_with_mock

    assert Dataset(name=random_hash(), asset_list=assets)


def test_cannot_add_assets_with_none_mock_to_dataset(
    asset_with_mock: dict[str, Any], asset_without_mock: dict[str, Any]
) -> None:
    dataset = Dataset(name=random_hash())

    with_mock = Asset(**asset_with_mock)
    with_none_mock = Asset(**asset_without_mock)

    dataset.add_asset(with_mock)

    with pytest.raises(ValueError) as excinfo:
        dataset.add_asset(with_none_mock)

    assert _ASSET_WITH_NONE_MOCK_ERROR_MESSAGE in str(excinfo.value)


def test_guest_client_get_empty_mock_as_private_pointer(
    worker: Worker,
    asset_with_empty_mock: dict[str, Any],
) -> None:
    asset = Asset(**asset_with_empty_mock)
    dataset = Dataset(name=random_hash(), asset_list=[asset])

    root_datasite_client = worker.root_client
    root_datasite_client.upload_dataset(dataset)

    guest_datasite_client = root_datasite_client.guest()
    guest_datasets = guest_datasite_client.api.services.dataset.get_all()
    guest_dataset = guest_datasets[0]

    mock = guest_dataset.assets[0].pointer

    assert not mock.is_real
    assert mock.is_pointer
    assert mock.syft_twin_type is TwinMode.MOCK


def test_datasite_client_cannot_upload_dataset_with_non_mock(worker: Worker) -> None:
    assets = [Asset(**make_asset_with_mock()) for _ in range(10)]
    dataset = Dataset(name=random_hash(), asset_list=assets)

    dataset.asset_list[0].mock = None

    root_datasite_client = worker.root_client

    with pytest.raises(ValueError) as excinfo:
        root_datasite_client.upload_dataset(dataset)

    assert _ASSET_WITH_NONE_MOCK_ERROR_MESSAGE in str(excinfo.value)


def test_adding_contributors_with_duplicate_email():
    # Datasets

    dataset = Dataset(name="Sample  dataset")
    res1 = dataset.add_contributor(
        role=sy.roles.UPLOADER, name="Alice", email="alice@naboo.net"
    )

    assert isinstance(res1, SyftSuccess)

    with pytest.raises(SyftException) as exc:
        dataset.add_contributor(
            role=sy.roles.UPLOADER, name="Alice Smith", email="alice@naboo.net"
        )
    assert exc.type == SyftException
    assert exc.value.public_message

    assert len(dataset.contributors) == 1

    # Assets
    asset = Asset(**make_asset_without_mock(), mock=ActionObject.empty())

    res3 = asset.add_contributor(
        role=sy.roles.UPLOADER, name="Bob", email="bob@naboo.net"
    )

    assert isinstance(res3, SyftSuccess)

    with pytest.raises(SyftException) as exc:
        asset.add_contributor(
            role=sy.roles.UPLOADER, name="Bob Abraham", email="bob@naboo.net"
        )

    assert exc.type == SyftException
    assert exc.value.public_message

    dataset.add_asset(asset)

    assert len(asset.contributors) == 1


@pytest.fixture(
    params=[
        1,
        "hello",
        {"key": "value"},
        {1, 2, 3},
        np.array([1, 2, 3]),
        pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}),
        torch.Tensor([1, 2, 3]),
    ]
)
def different_data_types(
    request,
) -> int | str | dict | set | np.ndarray | pd.DataFrame | torch.Tensor:
    return request.param


def test_upload_dataset_with_assets_of_different_data_types(
    worker: Worker,
    different_data_types: (
        int | str | dict | set | np.ndarray | pd.DataFrame | torch.Tensor
    ),
) -> None:
    asset = sy.Asset(
        name=random_hash(),
        data=different_data_types,
        mock=different_data_types,
    )
    dataset = Dataset(name=random_hash())
    dataset.add_asset(asset)
    root_datasite_client = worker.root_client
    res = root_datasite_client.upload_dataset(dataset)
    assert isinstance(res, SyftSuccess)
    assert len(root_datasite_client.api.services.dataset.get_all()) == 1
    assert type(root_datasite_client.datasets[0].assets[0].data) == type(
        different_data_types
    )
    assert type(root_datasite_client.datasets[0].assets[0].mock) == type(
        different_data_types
    )


def test_delete_small_datasets(worker: Worker, small_dataset: Dataset) -> None:
    root_client = worker.root_client
    assert not can_upload_to_blob_storage(small_dataset, root_client.metadata).unwrap()
    upload_res = root_client.upload_dataset(small_dataset)
    assert isinstance(upload_res, SyftSuccess)

    dataset = root_client.api.services.dataset.get_all()[0]
    asset = dataset.asset_list[0]
    assert isinstance(asset.data, np.ndarray)
    assert isinstance(asset.mock, np.ndarray)

    # delete the dataset without deleting its assets
    del_res = root_client.api.services.dataset.delete(
        uid=dataset.id, delete_assets=False
    )
    assert isinstance(del_res, SyftSuccess)
    assert isinstance(asset.data, np.ndarray)
    assert isinstance(asset.mock, np.ndarray)
    assert len(root_client.api.services.dataset.get_all()) == 0
    # we can still get back the deleted dataset by uid
    deleted_dataset = root_client.api.services.dataset.get_by_id(uid=dataset.id)
    assert deleted_dataset.name == f"_deleted_{dataset.name}_{dataset.id}"
    assert deleted_dataset.to_be_deleted

    # delete the dataset and its assets
    del_res = root_client.api.services.dataset.delete(
        uid=dataset.id, delete_assets=True
    )
    assert isinstance(del_res, SyftSuccess)
    assert asset.data is None
    with pytest.raises(SyftException):
        print(asset.mock)
    assert len(root_client.api.services.dataset.get_all()) == 0


def test_delete_big_datasets(worker: Worker, big_dataset: Dataset) -> None:
    root_client = worker.root_client
    assert can_upload_to_blob_storage(big_dataset, root_client.metadata).unwrap()
    upload_res = root_client.upload_dataset(big_dataset)
    assert isinstance(upload_res, SyftSuccess)

    dataset = root_client.api.services.dataset.get_all()[0]
    asset = dataset.asset_list[0]

    assert isinstance(asset.data, np.ndarray)
    assert isinstance(asset.mock, np.ndarray)
    # test that the data is saved in the blob storage
    assert len(root_client.api.services.blob_storage.get_all()) == 2

    # delete the dataset without deleting its assets
    del_res = root_client.api.services.dataset.delete(
        uid=dataset.id, delete_assets=False
    )
    assert isinstance(del_res, SyftSuccess)
    assert isinstance(asset.data, np.ndarray)
    assert isinstance(asset.mock, np.ndarray)
    assert len(root_client.api.services.dataset.get_all()) == 0
    # we can still get back the deleted dataset by uid
    deleted_dataset = root_client.api.services.dataset.get_by_id(uid=dataset.id)
    assert deleted_dataset.name == f"_deleted_{dataset.name}_{dataset.id}"
    assert deleted_dataset.to_be_deleted
    # the dataset's blob entries are still there
    assert len(root_client.api.services.blob_storage.get_all()) == 2

    # delete the dataset
    del_res = root_client.api.services.dataset.delete(
        uid=dataset.id, delete_assets=True
    )
    assert isinstance(del_res, SyftSuccess)
    assert asset.data is None
    with pytest.raises(SyftException):
        print(asset.mock)
    assert len(root_client.api.services.blob_storage.get_all()) == 0
    assert len(root_client.api.services.dataset.get_all()) == 0
