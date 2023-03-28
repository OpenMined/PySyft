# stdlib
from typing import List

# third party
import pytest

# syft absolute
from syft.core.node.new.dataset import Dataset
from syft.core.node.new.dataset_stash import ActionIDsPartitionKey
from syft.core.node.new.dataset_stash import NamePartitionKey
from syft.core.node.new.document_store import QueryKey
from syft.core.node.new.document_store import QueryKeys
from syft.core.node.new.uid import UID


def test_dataset_namepartitionkey():
    name_partition_key = QueryKeys(qks=NamePartitionKey).all[0]
    assert isinstance(name_partition_key, QueryKey)
    assert name_partition_key.key == "name"
    assert name_partition_key.type_ == str


def test_dataset_actionidpartitionkey():
    action_ids_partition_key = QueryKeys(qks=ActionIDsPartitionKey).all[0]
    assert isinstance(action_ids_partition_key, QueryKey)
    assert action_ids_partition_key.key == "action_ids"
    assert action_ids_partition_key.type_ == List[UID]


def test_dataset_get_by_name(mock_dataset_stash, mock_dataset):
    # retrieving existing dataset
    result = mock_dataset_stash.get_by_name(mock_dataset.name)
    assert result.is_ok(), f"Dataset could not be retrieved, result: {result}"
    assert isinstance(result.ok(), Dataset)
    assert result.ok().id == mock_dataset.id
    # retrieving non-existing dataset
    result = mock_dataset_stash.get_by_name("non_existing_dataset")
    assert result.is_ok(), f"Dataset could not be retrieved, result: {result}"
    assert result.ok() is None


@pytest.mark.xfail(
    raises=AttributeError,
    reason="DatasetUpdate is not implemeted yet",
)
def test_dataset_update(mock_dataset_stash, mock_dataset, mock_dataset_update):
    # succesful dataset update
    result = mock_dataset_stash.update(dataset_update=mock_dataset_update)
    assert result.is_ok(), f"Dataset could not be retrieved, result: {result}"
    assert isinstance(result.ok(), Dataset)
    assert mock_dataset.id == result.ok().id
    # error should be raised
    other_obj = object()
    result = mock_dataset_stash.update(dataset_update=other_obj)
    assert result.err(), (
        f"Dataset was updated with non-DatasetUpdate object," f"result: {result}"
    )


@pytest.mark.xfail(
    raises=(NotImplementedError, AssertionError),
    reason="StorePartition.find_index_or_search_keys is not implemeted yet",
)
def test_dataset_serach_action_ids(mock_dataset_stash, mock_dataset):
    # retrieving dataset by single action_id
    action_id = mock_dataset.id
    result = mock_dataset_stash.search_action_ids(uid=action_id)
    print(f"action_id: {action_id}; result: {result}; result.ok(): {result.ok()}")
    assert result.is_ok(), f"Dataset could not be retrieved, result: {result}"
    assert result.ok() != [], f"Dataset was not found by action_id {action_id}"
    assert isinstance(result.ok()[0], Dataset)
    assert result.ok().id == mock_dataset.id
    # retrieving dataset by list of action_ids
    result = mock_dataset_stash.search_action_ids(uid=[action_id])
    assert result.is_ok(), f"Dataset could not be retrieved, result: {result}"
    assert isinstance(result.ok()[0], Dataset)
    assert result.ok()[0].id == mock_dataset.id
    # retrieving dataset by non-existing action_id
    other_action_id = UID()
    result = mock_dataset_stash.search_action_ids(uid=other_action_id)
    assert result.is_ok(), f"Dataset could not be retrieved, result: {result}"
    assert result.ok() is None
