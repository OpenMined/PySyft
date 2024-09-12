# third party
import pytest

# syft absolute
from syft.service.dataset.dataset import Dataset
from syft.service.dataset.dataset_stash import DatasetStash
from syft.store.document_store_errors import NotFoundException
from syft.types.uid import UID


def test_dataset_get_by_name(
    root_verify_key, mock_dataset_stash: DatasetStash, mock_dataset: Dataset
) -> None:
    # retrieving existing dataset
    result = mock_dataset_stash.get_by_name(root_verify_key, mock_dataset.name)
    assert result.is_ok()
    assert isinstance(result.ok(), Dataset)
    assert result.ok().id == mock_dataset.id

    # retrieving non-existing dataset
    result = mock_dataset_stash.get_by_name(root_verify_key, "non_existing_dataset")
    assert result.is_err(), "Item not found"
    assert result.ok() is None
    assert type(result.err()) is NotFoundException


def test_dataset_search_action_ids(
    root_verify_key, mock_dataset_stash: DatasetStash, mock_dataset
):
    action_id = mock_dataset.assets[0].action_id

    result = mock_dataset_stash.search_action_ids(root_verify_key, uid=action_id)
    assert result.is_ok(), f"Dataset could not be retrieved, result: {result}"
    assert result.ok() != []
    assert isinstance(result.ok()[0], Dataset)
    assert result.ok()[0].id == mock_dataset.id

    # retrieving dataset by non-existing action_id
    other_action_id = UID()
    result = mock_dataset_stash.search_action_ids(root_verify_key, uid=other_action_id)
    assert result.is_ok()
    assert result.ok() == []

    # passing random object
    random_obj = object()
    with pytest.raises(ValueError):
        result = mock_dataset_stash.search_action_ids(root_verify_key, uid=random_obj)
