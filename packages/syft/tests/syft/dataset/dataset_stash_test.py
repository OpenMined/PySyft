# stdlib
from typing import List

# syft absolute
from syft.core.node.new.dataset import Asset
from syft.core.node.new.dataset import Dataset
from syft.core.node.new.dataset_stash import ActionIDsPartitionKey
from syft.core.node.new.dataset_stash import DatasetStash
from syft.core.node.new.dataset_stash import NamePartitionKey
from syft.core.node.new.document_store import BaseStash
from syft.core.node.new.document_store import QueryKey
from syft.core.node.new.document_store import QueryKeys
from syft.core.node.new.uid import UID


def test_dataset_namepartitionkey(dataset_stash):
    name_partition_key = QueryKeys(qks=NamePartitionKey).all[0]
    assert isinstance(name_partition_key, QueryKey) 
    assert name_partition_key.key == 'name'
    assert name_partition_key.type_ == str

def test_dataset_actionidpartitionkey(dataset_stash):
    action_ids_partition_key = QueryKeys(qks=ActionIDsPartitionKey).all[0]
    assert isinstance(action_ids_partition_key, QueryKey)
    assert action_ids_partition_key.key == "action_ids"
    assert action_ids_partition_key.type_ == List[UID]

def test_dataset_get_by_name(dataset_stash, empty_dataset):
    partitions = dataset_stash.set(empty_dataset)

    print(type(partitions), partitions)

    result = dataset_stash.get_by_name('empty_dataset')

    print(type(result), result)
    assert result.is_ok()
    assert isinstance(result, Dataset)
    

# def test_dataset_update():
#     pass

# def test_dataset_serach_action_ids():
#     pass