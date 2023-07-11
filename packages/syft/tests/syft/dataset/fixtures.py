# third party
import numpy as np
import pytest

# syft absolute
from syft.service.dataset.dataset import Asset
from syft.service.dataset.dataset import CreateAsset
from syft.service.dataset.dataset import Dataset
from syft.service.dataset.dataset import DatasetUpdate
from syft.service.dataset.dataset_stash import DatasetStash
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
    return DatasetStash(store=document_store)


@pytest.fixture
def mock_asset(worker, root_domain_client) -> Asset:
    # sometimes the access rights for client are overwritten
    # so we need to assing the root_client manually
    create_asset = CreateAsset(
        name="mock_asset",
        description="essential obj",
        data=np.array([0, 1, 2, 3, 4]),
        mock=np.array([0, 1, 1, 1, 1]),
        mock_is_real=False,
        node_uid=worker.id,
    )
    node_transform_context = TransformContext(
        node=worker,
        credentials=root_domain_client.credentials.verify_key,
        obj=create_asset,
    )
    mock_asset = create_asset.to(Asset, context=node_transform_context)
    return mock_asset


@pytest.fixture
def mock_dataset(root_verify_key, mock_dataset_stash, mock_asset) -> Dataset:
    mock_dataset = Dataset(id=UID(), name="test_dataset")
    mock_dataset.asset_list.append(mock_asset)
    result = mock_dataset_stash.partition.set(root_verify_key, mock_dataset)
    mock_dataset = result.ok()
    return mock_dataset


@pytest.fixture
def mock_dataset_update(mock_dataset):
    return DatasetUpdate()
