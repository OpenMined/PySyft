# stdlib
import random
from typing import Any
from uuid import uuid4

# third party
import numpy as np
from pydantic import ValidationError
import pytest

# syft absolute
import syft as sy
from syft.node.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.dataset.dataset import CreateAsset as Asset
from syft.service.dataset.dataset import CreateDataset as Dataset
from syft.service.dataset.dataset import _ASSET_WITH_NONE_MOCK_ERROR_MESSAGE
from syft.types.twin_object import TwinMode


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
    asset_without_mock: dict[str, Any]
):
    with pytest.raises(ValidationError):
        Asset(**asset_without_mock, mock_is_real=True)


def test_mock_always_not_real_after_calling_no_mock(
    asset_with_mock: dict[str, Any]
) -> None:
    asset = Asset(**asset_with_mock, mock_is_real=True)
    assert asset.mock_is_real

    asset.no_mock()
    assert not asset.mock_is_real


def test_mock_always_not_real_after_set_mock_to_empty(
    asset_with_mock: dict[str, Any]
) -> None:
    asset = Asset(**asset_with_mock, mock_is_real=True)
    assert asset.mock_is_real

    asset.no_mock()
    assert not asset.mock_is_real

    with pytest.raises(ValidationError):
        asset.mock_is_real = True

    asset.mock = mock()
    asset.mock_is_real = True
    assert asset.mock_is_real


def test_mock_always_not_real_after_set_to_empty(
    asset_with_mock: dict[str, Any]
) -> None:
    asset = Asset(**asset_with_mock, mock_is_real=True)
    assert asset.mock_is_real

    asset.mock = ActionObject.empty()
    assert not asset.mock_is_real

    with pytest.raises(ValidationError):
        asset.mock_is_real = True

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

    with pytest.raises(ValidationError):
        asset.set_mock(empty_mock, mock_is_real=True)

    assert asset.mock is asset_with_mock["mock"]


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

    root_domain_client = worker.root_client
    root_domain_client.upload_dataset(dataset)

    guest_domain_client = root_domain_client.guest()
    guest_datasets = guest_domain_client.api.services.dataset.get_all()
    guest_dataset = guest_datasets[0]

    mock = guest_dataset.assets[0].pointer

    assert not mock.is_real
    assert mock.is_pointer
    assert mock.syft_twin_type is TwinMode.MOCK


def test_domain_client_cannot_upload_dataset_with_non_mock(worker: Worker) -> None:
    assets = [Asset(**make_asset_with_mock()) for _ in range(10)]
    dataset = Dataset(name=random_hash(), asset_list=assets)

    dataset.asset_list[0].mock = None

    root_domain_client = worker.root_client

    with pytest.raises(ValueError) as excinfo:
        root_domain_client.upload_dataset(dataset)

    assert _ASSET_WITH_NONE_MOCK_ERROR_MESSAGE in str(excinfo.value)


def test_adding_contributors_with_duplicate_email():
    dataset = Dataset(name="dummy dataset")
    dataset.add_contributor(
        role=sy.roles.UPLOADER, name="Jim Carey", email="jim@69.com"
    )
    dataset.add_contributor(role=sy.roles.UPLOADER, name="Jim Car", email="jim@69.com")

    assert len(dataset.contributors) == 1
