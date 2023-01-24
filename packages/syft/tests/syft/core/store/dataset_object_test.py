# third party
import torch as th

# syft absolute
import syft as sy
from syft import serialize
from syft.core.common import UID
from syft.core.store.dataset import Dataset
from syft.core.store.storeable_object import StorableObject


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


def test_dataset_search_id() -> None:
    id = UID()
    data = UID()
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor n1"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([10, 20, 30, 40])
    description = "This is a dummy tensor n2"
    tags = ["dummy", "test"]
    obj3 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2, obj3]
    description = "This is a dataset"
    tags = ["dummy", "dataset"]
    dataset_obj = Dataset(id=id, data=data, description=description, tags=tags)

    assert dataset_obj.__contains__(_id=obj1.id)


def test_dataset_search_id_fail() -> None:
    id = UID()
    data = UID()
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor n1"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([10, 20, 30, 40])
    description = "This is a dummy tensor n2"
    tags = ["dummy", "test"]
    obj3 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2, obj3]
    description = "This is a dataset"
    tags = ["dummy", "dataset"]
    dataset_obj = Dataset(id=id, data=data, description=description, tags=tags)

    assert not dataset_obj.__contains__(_id=UID())


def test_dataset_get_element() -> None:
    id = UID()
    data = UID()
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor n1"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([10, 20, 30, 40])
    description = "This is a dummy tensor n2"
    tags = ["dummy", "test"]
    obj3 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2, obj3]
    description = "This is a dataset"
    tags = ["dummy", "dataset"]
    dataset_obj = Dataset(id=id, data=data, description=description, tags=tags)

    result = dataset_obj.__getitem__(_id=obj1.id)

    assert len(result) == 1
    assert result[0] == obj1


def test_dataset_get_element_fail() -> None:
    id = UID()
    data = UID()
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor n1"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([10, 20, 30, 40])
    description = "This is a dummy tensor n2"
    tags = ["dummy", "test"]
    obj3 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2, obj3]
    description = "This is a dataset"
    tags = ["dummy", "dataset"]
    dataset_obj = Dataset(id=id, data=data, description=description, tags=tags)

    assert dataset_obj.__getitem__(UID()) == []


def test_dataset_get_keys() -> None:
    id = UID()
    data = UID()
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor n1"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([10, 20, 30, 40])
    description = "This is a dummy tensor n2"
    tags = ["dummy", "test"]
    obj3 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2, obj3]
    description = "This is a dataset"
    tags = ["dummy", "dataset"]
    dataset_obj = Dataset(id=id, data=data, description=description, tags=tags)

    assert dataset_obj.keys() == [obj1.id, obj2.id, obj3.id]


def test_dataset_del() -> None:
    id = UID()
    data = UID()
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor n1"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([10, 20, 30, 40])
    description = "This is a dummy tensor n2"
    tags = ["dummy", "test"]
    obj3 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2, obj3]
    description = "This is a dataset"
    tags = ["dummy", "dataset"]
    dataset_obj = Dataset(id=id, data=data, description=description, tags=tags)

    dataset_obj.__delitem__(obj2.id)

    assert dataset_obj.data == [obj1, obj3]


def test_serde_storable_obj() -> None:
    id = UID()
    data = th.Tensor([-1, -2, -3, -4])
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor n1"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([10, 20, 30, 40])
    description = "This is a dummy tensor n2"
    tags = ["dummy", "test"]
    obj3 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2, obj3]
    description = "This is a dataset"
    tags = ["dummy", "dataset"]
    dataset_obj = Dataset(id=id, data=data, description=description, tags=tags)

    blob = sy.serialize(obj=dataset_obj)
    sy.deserialize(blob=blob)


def test_serde_storable_obj_2() -> None:
    id = UID()
    data = th.Tensor([-1, -2, -3, -4])
    description = "This is a dummy id"
    tags = ["dummy", "test"]
    obj1 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([1, 2, 3, 4])
    description = "This is a dummy tensor n1"
    tags = ["dummy", "test"]
    obj2 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = th.Tensor([10, 20, 30, 40])
    description = "This is a dummy tensor n2"
    tags = ["dummy", "test"]
    obj3 = StorableObject(id=id, data=data, description=description, tags=tags)

    id = UID()
    data = [obj1, obj2, obj3]
    description = "This is a dataset"
    tags = ["dummy", "dataset"]
    obj = Dataset(id=id, data=data, description=description, tags=tags)

    blob = serialize(obj)
    ds_obj = sy.deserialize(blob=blob)

    assert obj.id == ds_obj.id
    assert obj.description == ds_obj.description
    assert obj.tags == ds_obj.tags
    assert len(obj.data) == len(ds_obj.data)

    for i in range(len(obj.data)):
        assert obj.data[i].id == ds_obj.data[i].id
        assert th.all(th.eq(obj.data[i].data, ds_obj.data[i].data))
        assert obj.data[i].description == ds_obj.data[i].description
        assert obj.data[i].tags == ds_obj.data[i].tags
