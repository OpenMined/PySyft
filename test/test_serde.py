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

    def test_ndarray_simplify(self):
        input = numpy.random.random((100, 100))
        output = _simplify(input)
        assert type(output[1]) == bytes


class TestSerde(TestCase):
    def test_torch_tensor(self):
        t = tensor(numpy.random.random((100, 100)))
        t_serialized = serialize(t, compress=False)
        t_serialized_deserialized = deserialize(t_serialized, compressed=False)
        assert (t == t_serialized_deserialized).all()

    def test_tuple(self):
        # Test with a simple datatype
        tuple = (1, 2)
        tuple_serialized = serialize(tuple, compress=False)
        tuple_serialized_deserialized = deserialize(tuple_serialized, compressed=False)
        assert tuple == tuple_serialized_deserialized

        # Test with a complex data structure
        tensor_one = tensor(numpy.random.random((100, 100)))
        tensor_two = tensor(numpy.random.random((100, 100)))
        tuple = (tensor_one, tensor_two)
        tuple_serialized = serialize(tuple, compress=False)
        tuple_serialized_deserialized = deserialize(tuple_serialized, compressed=False)
        # `assert tuple_serialized_deserialized == tuple` does not work, therefore it's split into 3 assertions
        assert type(tuple_serialized_deserialized) == type(tuple)
        assert (tuple_serialized_deserialized[0] == tensor_one).all()
        assert (tuple_serialized_deserialized[1] == tensor_two).all()

    def test_bytearray(self):
        bytearr = bytearray("This is a teststring", "utf-8")
        bytearr_serialized = serialize(bytearr, compress=False)
        bytearr_serialized_desirialized = deserialize(
            bytearr_serialized, compressed=False
        )
        assert bytearr == bytearr_serialized_desirialized

        bytearr = bytearray(numpy.random.random((100, 100)))
        bytearr_serialized = serialize(bytearr, compress=False)
        bytearr_serialized_desirialized = deserialize(
            bytearr_serialized, compressed=False
        )
        assert bytearr == bytearr_serialized_desirialized

    def test_ndarray_serde(self):
        arr = numpy.random.random((100, 100))
        arr_serialized = serialize(arr, compress=False)

        arr_serialized_deserialized = deserialize(arr_serialized, compressed=False)
        assert numpy.array_equal(arr,arr_serialized_deserialized)

