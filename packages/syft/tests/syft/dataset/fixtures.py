# third party
import pytest

# syft absolute
from syft.core.node.new.dataset import Dataset
from syft.core.node.new.dataset_stash import DatasetStash
from syft.core.node.new.uid import UID


@pytest.fixture
def dataset_stash(document_store) -> DatasetStash:
    return DatasetStash(store=document_store)


@pytest.fixture
def empty_dataset(dataset_stash):
    return Dataset(
        id=UID(),
        name="test_dataset",
    )
