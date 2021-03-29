# stdlib
from collections import OrderedDict as PyOrderectDict

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.lib.python.collections.ordered_dict import OrderedDict
from syft.lib.python.int import Int
from syft.lib.python.string import String
from syft.proto.lib.python.collections.ordered_dict_pb2 import (
    OrderedDict as OrderedDict_PB,
)


def test_dict_creation() -> None:
    d1 = {String("t1"): 1, String("t2"): 2}
    dict1 = OrderedDict(d1)
    assert type(getattr(dict1, "id", None)) is UID

    d2 = dict({"t1": 1, "t2": 2})
    dict2 = OrderedDict(d2)
    dict2._id = UID()
    assert type(getattr(dict2, "id", None)) is UID

    d3 = OrderedDict({"t1": 1, "t2": 2})
    dict3 = OrderedDict(d3)
    assert type(getattr(dict3, "id", None)) is UID

    assert dict1.keys() == dict2.keys()
    assert dict1.keys() == dict3.keys()


def test_dict_serde() -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = OrderedDict({Int(1): t1, Int(2): t2})
    assert type(getattr(syft_list, "id", None)) is UID

    serialized = syft_list._object2proto()

    assert isinstance(serialized, OrderedDict_PB)

    deserialized = OrderedDict._proto2object(proto=serialized)

    assert isinstance(deserialized, OrderedDict)
    assert isinstance(deserialized, PyOrderectDict)

    assert deserialized.id == syft_list.id
    for deserialized_el, original_el in zip(deserialized, syft_list):
        assert deserialized_el == original_el


def test_list_send(root_client: sy.VirtualMachineClient) -> None:
    syft_list = OrderedDict(
        {String("t1"): String("test"), String("t2"): String("test")}
    )
    ptr = syft_list.send(root_client)
    # Check pointer type
    assert ptr.__class__.__name__ == "OrderedDictPointer"

    # Check that we can get back the object
    res = ptr.get()
    for res_el, original_el in zip(res, syft_list):
        assert res_el == original_el


@pytest.mark.parametrize("method_name", ["items", "keys", "values"])
def test_iterator_methods(
    method_name: str, root_client: sy.VirtualMachineClient
) -> None:
    d = OrderedDict({"#1": 1, "#2": 2})
    dptr = d.send(root_client)

    itemsptr = getattr(dptr, method_name)()
    assert type(itemsptr).__name__ == "IteratorPointer"

    for itemptr, local_item in zip(itemsptr, getattr(d, method_name)()):
        get_item = itemptr.get()
        assert get_item == local_item
