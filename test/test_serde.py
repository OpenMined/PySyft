from syft.serde import _simplify
from syft.serde import serialize
from syft.serde import deserialize
from unittest import TestCase
from torch import tensor
import numpy


class TestSimplify(TestCase):
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

    def test_range_simplify(self):
        input = range(1, 3, 4)
        target = [5, [1, 3, 4]]
        assert _simplify(input) == target

    def test_torch_tensor_simplify(self):
        input = tensor(numpy.random.random((100, 100)))
        output = _simplify(input)
        assert type(output) == list
        assert type(output[1]) == bytes


class TestSerde(TestCase):
    def test_torch_tensor_serde(self):
        t = tensor(numpy.random.random((100, 100)))
        t_serialized = serialize(t, compress=False)
        t_serialized_deserialized = deserialize(t_serialized, compressed=False)
        assert (t == t_serialized_deserialized).all()
