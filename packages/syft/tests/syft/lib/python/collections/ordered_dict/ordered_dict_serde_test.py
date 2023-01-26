# stdlib
from collections import OrderedDict as PyOrderectDict

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft import deserialize
from syft import serialize
from syft.lib.python.collections.ordered_dict import SyOrderedDict
from syft.lib.python.int import Int
from syft.lib.python.string import String


def test_dict_creation() -> None:
    d1 = {String("t1"): 1, String("t2"): 2}
    dict1 = SyOrderedDict(d1)

    d2 = dict({"t1": 1, "t2": 2})
    dict2 = SyOrderedDict(d2)

    d3 = SyOrderedDict({"t1": 1, "t2": 2})
    dict3 = SyOrderedDict(d3)

    assert dict1.keys() == dict2.keys()
    assert dict1.keys() == dict3.keys()


def test_dict_serde() -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = SyOrderedDict({Int(1): t1, Int(2): t2})

    serialized = sy.serialize(syft_list)

    deserialized = sy.deserialize(serialized)

    assert isinstance(deserialized, SyOrderedDict)
    assert isinstance(deserialized, PyOrderectDict)

    for deserialized_el, original_el in zip(deserialized, syft_list):
        assert deserialized_el == original_el


def test_dict_serde_bytes() -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = SyOrderedDict({Int(1): t1, Int(2): t2})

    serialized = serialize(syft_list, to_bytes=True)

    assert isinstance(serialized, bytes)

    deserialized = deserialize(serialized, from_bytes=True)

    assert isinstance(deserialized, SyOrderedDict)
    assert isinstance(deserialized, PyOrderectDict)

    for deserialized_el, original_el in zip(deserialized, syft_list):
        assert deserialized_el == original_el


def test_list_send(root_client: sy.VirtualMachineClient) -> None:
    syft_list = SyOrderedDict(
        {String("t1"): String("test"), String("t2"): String("test")}
    )
    ptr = syft_list.send(root_client)

    # Check pointer type
    assert ptr.__class__.__name__ == "SyOrderedDictPointer"

    # Check that we can get back the object
    res = ptr.get()
    for res_el, original_el in zip(res, syft_list):
        assert res_el == original_el


# MADHAVA: this needs fixing
@pytest.mark.xfail
@pytest.mark.parametrize("method_name", ["items", "keys", "values"])
def test_iterator_methods(
    method_name: str, root_client: sy.VirtualMachineClient
) -> None:
    d = SyOrderedDict({"#1": 1, "#2": 2})
    dptr = d.send(root_client)

    itemsptr = getattr(dptr, method_name)()
    assert type(itemsptr).__name__ == "IteratorPointer"

    for itemptr, local_item in zip(itemsptr, getattr(d, method_name)()):
        get_item = itemptr.get()
        assert get_item == local_item


def test_ordered_dict_bytes() -> None:
    # Testing if multiple serialization of the similar object results in same bytes
    d1 = {String("t1"): 1, String("t2"): 2}
    dict1 = SyOrderedDict(d1)
    dict2 = SyOrderedDict(d1)
    assert sy.serialize(dict1, to_bytes=True) == sy.serialize(dict2, to_bytes=True)
