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
from syft.core.node.new.uid import UID


@pytest.fixture
def mock_dataset_stash(document_store) -> DatasetStash:
    return DatasetStash(store=document_store)


@pytest.fixture
def mock_asset(action_store, root_domain_client) -> Asset:
    create_asset = CreateAsset(
        name="mock_asset",
        description="essential obj",
        data=np.array([0, 1, 2, 3, 4]),
        mock=[0, 1],
        mock_is_real=False,
    )

    # mock_asset = create_asset.to(Asset)
    asset_uid = UID()
    action_store.set(
        uid=asset_uid,
        credentials=root_domain_client.credentials,
        syft_object=create_asset,
    )
    resoponse = action_store.get(
        uid=asset_uid, credentials=root_domain_client.credentials
    )
    return resoponse.ok()


@pytest.fixture
def mock_dataset(root_domain_client, mock_dataset_stash, mock_asset) -> Dataset:
    mock_dataset = CreateDataset(id=UID(), name="test_dataset")
    # print(type(mock_dataset), mock_dataset, mock_dataset.assets)
    mock_dataset.add_asset(mock_asset)
    # print(type(mock_dataset), mock_dataset, mock_dataset.assets)
    root_domain_client.upload_dataset(mock_dataset)
    # mock_dataset = root_domain_client.datasets[0]
    return mock_dataset


# @pytest.fixture
# def mock_dataset(root_domain_client, mock_dataset_stash, mock_asset) -> Dataset:
#     mock_dataset = Dataset(id=UID(), name="test_dataset")
#     result = mock_dataset_stash.partition.set(mock_dataset)
#     # root_domain_client.upload_dataset(mock_dataset)
#     # mock_dataset = root_domain_client.api.services.dataset.get_all()
#     # print(mock_dataset[0])
#     mock_dataset = result.ok()
#     return mock_dataset


@pytest.fixture
def mock_dataset_update(mock_dataset):
    return DatasetUpdate()
