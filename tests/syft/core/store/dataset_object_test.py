# third party
import torch as th

# syft absolute
import syft as sy
from syft.core.common import UID
from syft.core.store.storeable_object import StorableObject
from syft.core.store.dataset import Dataset


def test_create_dataset_obj_with_id() -> None:
    id = UID()
    data = [UID(), UID()]
    description = "This is a dummy test"
    tags = ["dummy", "test"]
    Dataset(id=id, data=data, description=description, tags=tags)


def test_create_dataset_with_store_obj() -> None:
    id = UID()
    data = UID()
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2]
    description = "This is a dummy tensor"
    tags = ["dummy", "dataset"]
    Dataset(id=id, data=data, description=description, tags=tags)
