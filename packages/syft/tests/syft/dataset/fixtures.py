# third party
import pytest

# syft absolute
from syft.core.node.new.dataset import Dataset
from syft.core.node.new.dataset import DatasetUpdate
from syft.core.node.new.dataset_stash import DatasetStash
from syft.core.node.new.uid import UID


@pytest.fixture
def mock_dataset_stash(document_store) -> DatasetStash:
    return DatasetStash(store=document_store)


@pytest.fixture
def mock_dataset(mock_dataset_stash):
    mock_dataset = Dataset(id=UID(), name="test_dataset")
    result = mock_dataset_stash.partition.set(mock_dataset)
    return result.ok()


@pytest.fixture
def mock_dataset_update(mock_dataset):
    return DatasetUpdate()
