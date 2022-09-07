# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray


def create_test_dataset(client, name: str = "TSTDataset"):
    data = np.array([1, 2, 3, 4, 5])
    data_subject_name = "testing"
    entities = np.broadcast_to(
        np.array(DataSubjectArray([data_subject_name])), data.shape
    )

    train_data = sy.Tensor(data).private(min_val=0, max_val=255, data_subjects=entities)
    test_data = sy.Tensor(data).private(min_val=0, max_val=255, data_subjects=entities)
    val_data = sy.Tensor(data).private(min_val=0, max_val=255, data_subjects=entities)
    client.load_dataset(
        name=name,
        assets={
            "train_data": train_data,
            "test_data": test_data,
            "val_data": val_data,
        },
        description="Test data",
        skip_checks=True,
    )


@pytest.mark.redis
def test_create_dataset(domain_owner, cleanup_storage):
    assert len(domain_owner.datasets) == 0

    create_test_dataset(domain_owner)

    assert len(domain_owner.datasets) == 1
    assert len(domain_owner.datasets[0].data) == 3


@pytest.mark.redis
def test_delete_dataset_assets(domain_owner, cleanup_storage):

    create_test_dataset(domain_owner)

    # Check if the dataset has been loaded
    assert len(domain_owner.datasets) == 1
    assert len(domain_owner.datasets[0].data) == 3

    # Delete train_data asset
    domain_owner.datasets[0].delete("train_data", skip_check=True)

    # Check if the number of assets has been decreased
    assert len(domain_owner.datasets[0].data) == 2

    # Check if the asset key (train_data) is still there
    for asset in domain_owner.datasets[0].data:
        assert asset["name"] != "train_data"

    domain_owner.datasets[0].delete("test_data", skip_check=True)

    # Check if the number of assets has been decreased
    assert len(domain_owner.datasets[0].data) == 1

    # Check if the asset key (train_data) is still there
    for asset in domain_owner.datasets[0].data:
        if asset["name"] == "test_data":
            raise Exception

    domain_owner.datasets[0].delete("val_data", skip_check=True)

    # Check if the number of assets has been decreased
    assert len(domain_owner.datasets[0].data) == 0

    # Check if the asset key (train_data) is still there
    for asset in domain_owner.datasets[0].data:
        assert asset["name"] != "val_data"


@pytest.mark.redis
def test_delete_entire_dataset(domain_owner, cleanup_storage):

    create_test_dataset(domain_owner, "Dataset_1")
    create_test_dataset(domain_owner, "Dataset_2")

    # Check if the number of datasets loaded
    assert len(domain_owner.datasets) == 2
    assert len(domain_owner.datasets[0].data) == 3
    assert len(domain_owner.datasets[1].data) == 3

    assert domain_owner.datasets[0].name == "Dataset_1"
    assert domain_owner.datasets[1].name == "Dataset_2"

    domain_owner.datasets.delete(dataset_id=domain_owner.datasets[0].id)

    # Check if the number of available datasets has been decreased
    assert len(domain_owner.datasets) == 1

    # Check if the Dataset_2 is still there
    assert domain_owner.datasets[0].name == "Dataset_2"


@pytest.mark.redis
def test_purge_datasets(domain_owner, cleanup_storage):
    create_test_dataset(domain_owner, "Dataset_1")
    create_test_dataset(domain_owner, "Dataset_2")
    create_test_dataset(domain_owner, "Dataset_3")
    create_test_dataset(domain_owner, "Dataset_4")

    # Check if the number of datasets loaded
    assert len(domain_owner.datasets) == 4

    domain_owner.datasets.purge(skip_check=True)

    # Check if the number of available datasets has been decreased
    assert len(domain_owner.datasets) == 0
