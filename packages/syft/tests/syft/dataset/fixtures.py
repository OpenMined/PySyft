# third party
import numpy as np
import pytest

# syft absolute
from syft.service.dataset.dataset import Asset
from syft.service.dataset.dataset import Contributor
from syft.service.dataset.dataset import CreateAsset
from syft.service.dataset.dataset import Dataset
from syft.service.dataset.dataset import DatasetUpdate
from syft.service.dataset.dataset_stash import DatasetStash
from syft.service.user.roles import Roles
from syft.types.transforms import TransformContext
from syft.types.uid import UID


def create_asset() -> CreateAsset:
    return CreateAsset(
        name="mock_asset",
        description="essential obj",
        data=np.array([0, 1, 2, 3, 4]),
        mock=np.array([0, 1, 1, 1, 1]),
        mock_is_real=False,
    )


@pytest.fixture
def mock_dataset_stash(document_store) -> DatasetStash:
    yield DatasetStash(store=document_store)


@pytest.fixture
def mock_asset(worker, root_datasite_client) -> Asset:
    # sometimes the access rights for client are overwritten
    # so we need to assing the root_client manually
    uploader = Contributor(
        role=str(Roles.UPLOADER),
        name="test",
        email="test@test.org",
    )
    create_asset = CreateAsset(
        name="mock_asset",
        description="essential obj",
        data=np.array([0, 1, 2, 3, 4]),
        mock=np.array([0, 1, 1, 1, 1]),
        mock_is_real=False,
        server_uid=worker.id,
        uploader=uploader,
        contributors=[uploader],
        syft_server_location=worker.id,
        syft_client_verify_key=root_datasite_client.credentials.verify_key,
    )
    server_transform_context = TransformContext(
        server=worker,
        credentials=root_datasite_client.credentials.verify_key,
        obj=create_asset,
    )
    mock_asset = create_asset.to(Asset, context=server_transform_context)
    yield mock_asset


@pytest.fixture
def mock_dataset(
    root_verify_key, mock_dataset_stash: DatasetStash, mock_asset
) -> Dataset:
    uploader = Contributor(
        role=str(Roles.UPLOADER),
        name="test",
        email="test@test.org",
    )
    mock_dataset = Dataset(
        id=UID(), name="test_dataset", uploader=uploader, contributors=[uploader]
    )
    mock_dataset.asset_list.append(mock_asset)
    result = mock_dataset_stash.set(root_verify_key, mock_dataset)
    mock_dataset = result.ok()
    yield mock_dataset


@pytest.fixture
def mock_dataset_update(mock_dataset):
    yield DatasetUpdate()
