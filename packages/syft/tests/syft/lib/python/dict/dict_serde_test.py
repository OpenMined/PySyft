# stdlib
from collections import UserDict

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.lib.python.dict import Dict
from syft.lib.python.int import Int
from syft.lib.python.string import String


def test_dict_creation() -> None:
    d1 = {String("t1"): 1, String("t2"): 2}
    dict1 = Dict(d1)

    d2 = dict({"t1": 1, "t2": 2})
    dict2 = Dict(d2)

    d3 = UserDict({"t1": 1, "t2": 2})
    dict3 = Dict(**d3)

    assert dict1.keys() == dict2.keys()
    assert dict1.keys() == dict3.keys()

    # ValuesView uses object.__eq__
    # https://stackoverflow.com/questions/34312674/why-are-the-values-of-an-ordereddict-not-equal
    assert dict1.values() != dict2.values()
    assert dict1.values() != dict3.values()

    assert dict1.items() == dict2.items()
    assert dict1.items() == dict3.items()

    it = list(iter(dict2.values()))
    assert len(it) == 2
    assert type(it) is list


def test_dict_serde() -> None:
    t1 = th.tensor([1, 2])
    t2 = th.tensor([1, 3])

    syft_list = Dict({Int(1): t1, Int(2): t2})

    serialized = sy.serialize(syft_list)

    deserialized = sy.deserialize(serialized)

    assert isinstance(deserialized, Dict)
    for deserialized_el, original_el in zip(deserialized, syft_list):
        assert deserialized_el == original_el


def test_list_send(client: sy.VirtualMachineClient) -> None:
    syft_list = Dict({String("t1"): String("test"), String("t2"): String("test")})
    ptr = syft_list.send(client)
    # Check pointer type
    assert ptr.__class__.__name__ == "DictPointer"

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
    d = Dict({"#1": 1, "#2": 2})
    dptr = d.send(root_client)

    itemsptr = getattr(dptr, method_name)()
    assert type(itemsptr).__name__ == "IteratorPointer"

    for itemptr, local_item in zip(itemsptr, getattr(d, method_name)()):
        get_item = itemptr.get()
        assert get_item == local_item


def test_dict_bytes() -> None:
    # Testing if multiple serialization of the similar object results in same bytes
    value_1 = Dict({"Hello": "OM"})
    value_2 = Dict({"Hello": "OM"})
    assert sy.serialize(value_1, to_bytes=True) == sy.serialize(value_2, to_bytes=True)
