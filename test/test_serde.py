from syft.serde import _simplify, serialize, deserialize
from unittest import TestCase
from torch import tensor
import numpy
import msgpack


class TupleSerde(TestCase):
    def test_tuple_simplify(self):
        input = ("hello", "world")
        target = [1, ("hello", "world")]
        assert _simplify(input) == target

    def test_list_simplify(self):
        input = ["hello", "world"]
        target = [2, ["hello", "world"]]
        assert _simplify(input) == target

    def test_set_simplify(self):
        input = set(["hello", "world"])
        target = [3, set(["hello", "world"])]
        assert _simplify(input) == target

    def test_float_simplify(self):
        input = 5.6
        target = 5.6
        assert _simplify(input) == target

    def test_int_simplify(self):
        input = 5
        target = 5
        assert _simplify(input) == target

    def test_string_simplify(self):
        input = "hello"
        target = "hello"
        assert _simplify(input) == target

    def test_dict_simplify(self):
        input = {"hello": "world"}
        target = [4, {"hello": "world"}]
        assert _simplify(input) == target

    def test_torch_tensor_serde(self):
        t = tensor(numpy.random.random((100, 100)))
        t_serialized = serialize(t, compress=False)
        t_serialized_deserialized = deserialize(t_serialized, compressed=False)
        assert (t == t_serialized_deserialized).all()

    def test_tuple_serde(self):
        tuple = (1,2)
        tuple_serialized = serialize(tuple,compress=False)
        tuple_deserialized = deserialize(tuple_serialized, compressed=False)
        assert tuple == tuple_deserialized
