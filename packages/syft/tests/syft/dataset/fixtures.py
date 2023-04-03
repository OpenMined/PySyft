# third party
import numpy as np
import pytest

# syft absolute
from syft.core.node.new.dataset import Asset
from syft.core.node.new.dataset import CreateAsset
from syft.core.node.new.dataset import CreateDataset
from syft.core.node.new.dataset import Dataset
from syft.core.node.new.dataset import DatasetUpdate
from syft.core.node.new.dataset_stash import DatasetStash
from syft.core.node.new.transforms import TransformContext
from syft.core.node.new.uid import UID


def create_asset() -> CreateAsset:
    return CreateAsset(
        name="mock_asset",
        description="essential obj",
        data=np.array([0, 1, 2, 3, 4]),
        mock=np.array([0, 1, 1, 1, 1]),
        mock_is_real=False
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
        node_uid=worker.id
    )
    node_transform_context = TransformContext(
        node=worker,
        credentials=root_domain_client.credentials.verify_key,
        obj=create_asset
    )
    mock_asset = create_asset.to(Asset, context=node_transform_context)
    return mock_asset

@pytest.fixture
def mock_dataset(mock_dataset_stash, mock_asset) -> Dataset:
    mock_dataset = Dataset(id=UID(), name="test_dataset")
    mock_dataset.asset_list.append(mock_asset)
    result = mock_dataset_stash.partition.set(mock_dataset)
    mock_dataset = result.ok()
    return mock_dataset


@pytest.fixture
def mock_dataset_update(mock_dataset):
    return DatasetUpdate()
