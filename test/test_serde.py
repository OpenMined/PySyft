from syft.serde import _simplify
from syft.serde import serialize
from syft.serde import deserialize
from syft.serde import _compress
from syft.serde import _decompress
from syft.frameworks.torch.tensors import PointerTensor
import torch
import syft
from unittest import TestCase
from torch import Tensor
import numpy
import msgpack


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
        target = [3, ["hello", "world"]]
        assert _simplify(input)[0] == target[0]
        assert set(_simplify(input)[1]) == set(target[1])

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
        target = [5, (1, 3, 4)]
        assert _simplify(input) == target

    def test_torch_tensor_simplify(self):
        input = Tensor(numpy.random.random((100, 100)))
        output = _simplify(input)
        assert type(output) == list
        assert type(output[1]) == bytes

    def test_ndarray_simplify(self):
        input = numpy.random.random((100, 100))
        output = _simplify(input)
        assert type(output[1][0]) == bytes
        assert output[1][1] == input.shape
        assert output[1][2] == input.dtype.name

    def test_pointer_tensor_simplify(self):
        alice = syft.VirtualWorker(id="alice")
        input = PointerTensor(id=1000, location=alice, owner=alice)
        output = _simplify(input)
        assert output[1]["id"] == input.id
        assert output[1]["owner"] == input.owner.id
        assert output[1]["location"] == input.location.id
        assert output[1]["id_at_location"] == input.id_at_location


class TestSerde(TestCase):
    def setUp(self):
        hook = syft.TorchHook(torch, verbose=True)

        me = hook.local_worker
        me.is_client_worker = False

        bob = syft.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
        alice = syft.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
        james = syft.VirtualWorker(id="james", hook=hook, is_client_worker=False)

        me.add_workers([bob, alice, james])
        bob.add_workers([alice, james])
        alice.add_workers([bob, james])
        james.add_workers([bob, alice])

        self.hook = hook
        self.bob = bob
        self.alice = alice
        self.james = james

    def test_torch_Tensor(self):
        t = Tensor(numpy.random.random((100, 100)))
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
        tensor_one = Tensor(numpy.random.random((100, 100)))
        tensor_two = Tensor(numpy.random.random((100, 100)))
        tuple = (tensor_one, tensor_two)
        tuple_serialized = serialize(tuple, compress=False)
        tuple_serialized_deserialized = deserialize(tuple_serialized, compressed=False)
        # `assert tuple_serialized_deserialized == tuple` does not work, therefore it's split
        # into 3 assertions
        assert type(tuple_serialized_deserialized) == type(tuple)
        assert (tuple_serialized_deserialized[0] == tensor_one).all()
        assert (tuple_serialized_deserialized[1] == tensor_two).all()

    def test_compressed_tuple(self):
        # Test with a simple datatype
        tuple = (1, 2)
        tuple_serialized = serialize(tuple, compress=True)
        tuple_serialized_deserialized = deserialize(tuple_serialized, compressed=True)
        assert tuple == tuple_serialized_deserialized

        # Test with a complex data structure
        tensor_one = Tensor(numpy.random.random((100, 100)))
        tensor_two = Tensor(numpy.random.random((100, 100)))
        tuple = (tensor_one, tensor_two)
        tuple_serialized = serialize(tuple, compress=True)
        tuple_serialized_deserialized = deserialize(tuple_serialized, compressed=True)
        # `assert tuple_serialized_deserialized == tuple` does not work, therefore it's split
        # into 3 assertions
        assert type(tuple_serialized_deserialized) == type(tuple)
        assert (tuple_serialized_deserialized[0] == tensor_one).all()
        assert (tuple_serialized_deserialized[1] == tensor_two).all()

    def test_bytearray(self):
        bytearr = bytearray("This is a teststring", "utf-8")
        bytearr_serialized = serialize(bytearr, compress=False)
        bytearr_serialized_desirialized = deserialize(bytearr_serialized, compressed=False)
        assert bytearr == bytearr_serialized_desirialized

        bytearr = bytearray(numpy.random.random((100, 100)))
        bytearr_serialized = serialize(bytearr, compress=False)
        bytearr_serialized_desirialized = deserialize(bytearr_serialized, compressed=False)
        assert bytearr == bytearr_serialized_desirialized

    def test_compressed_bytearray(self):
        bytearr = bytearray("This is a teststring", "utf-8")
        bytearr_serialized = serialize(bytearr, compress=True)
        bytearr_serialized_desirialized = deserialize(bytearr_serialized, compressed=True)
        assert bytearr == bytearr_serialized_desirialized

        bytearr = bytearray(numpy.random.random((100, 100)))
        bytearr_serialized = serialize(bytearr, compress=True)
        bytearr_serialized_desirialized = deserialize(bytearr_serialized, compressed=True)
        assert bytearr == bytearr_serialized_desirialized

    def test_ndarray_serde(self):
        arr = numpy.random.random((100, 100))
        arr_serialized = serialize(arr, compress=False)

        arr_serialized_deserialized = deserialize(arr_serialized, compressed=False)

        assert numpy.array_equal(arr, arr_serialized_deserialized)

    def test_compress_decompress_lz4(self):
        original = msgpack.dumps([1, 2, 3])
        compressed = _compress(original)
        decompressed = _decompress(compressed)
        assert type(compressed) == bytes
        assert original == decompressed

    def test_compress_decompress_zstd(self):
        original = msgpack.dumps([1, 2, 3])
        compressed = _compress(original, "zstd")
        decompressed = _decompress(compressed, "zstd")
        assert type(compressed) == bytes
        assert original == decompressed

    def test_compressed_serde_lz4(self):
        arr = numpy.random.random((100, 100))
        arr_serialized = serialize(arr, compress=True)

        arr_serialized_deserialized = deserialize(arr_serialized, compressed=True)
        assert numpy.array_equal(arr, arr_serialized_deserialized)

    def test_compressed_serde_zstd(self):
        arr = numpy.random.random((100, 100))
        arr_serialized = serialize(arr, compress=True, compressScheme="zstd")
        arr_serialized_deserialized = deserialize(
            arr_serialized, compressed=True, compressScheme="zstd"
        )
        assert numpy.array_equal(arr, arr_serialized_deserialized)

    def test_dict(self):
        # Test with integers
        _dict = {1: 1, 2: 2, 3: 3}
        dict_serialized = serialize(_dict, compress=False)
        dict_serialized_deserialized = deserialize(dict_serialized, compressed=False)
        assert _dict == dict_serialized_deserialized

        # Test with strings
        _dict = {"one": 1, "two": 2, "three": 3}
        dict_serialized = serialize(_dict, compress=False)
        dict_serialized_deserialized = deserialize(dict_serialized, compressed=False)
        assert _dict == dict_serialized_deserialized

        # Test with a complex data structure
        tensor_one = Tensor(numpy.random.random((100, 100)))
        tensor_two = Tensor(numpy.random.random((100, 100)))
        _dict = {0: tensor_one, 1: tensor_two}
        dict_serialized = serialize(_dict, compress=False)
        dict_serialized_deserialized = deserialize(dict_serialized, compressed=False)
        # `assert dict_serialized_deserialized == _dict` does not work, therefore it's split
        # into 3 assertions
        assert type(dict_serialized_deserialized) == type(_dict)
        assert (dict_serialized_deserialized[0] == tensor_one).all()
        assert (dict_serialized_deserialized[1] == tensor_two).all()

    def test_compressed_dict(self):
        # Test with integers
        _dict = {1: 1, 2: 2, 3: 3}
        dict_serialized = serialize(_dict, compress=True)
        dict_serialized_deserialized = deserialize(dict_serialized, compressed=True)
        assert _dict == dict_serialized_deserialized

        # Test with strings
        _dict = {"one": 1, "two": 2, "three": 3}
        dict_serialized = serialize(_dict, compress=True)
        dict_serialized_deserialized = deserialize(dict_serialized, compressed=True)
        assert _dict == dict_serialized_deserialized

        # Test with a complex data structure
        tensor_one = Tensor(numpy.random.random((100, 100)))
        tensor_two = Tensor(numpy.random.random((100, 100)))
        _dict = {0: tensor_one, 1: tensor_two}
        dict_serialized = serialize(_dict, compress=True)
        dict_serialized_deserialized = deserialize(dict_serialized, compressed=True)
        # `assert dict_serialized_deserialized == _dict` does not work, therefore it's split
        # into 3 assertions
        assert type(dict_serialized_deserialized) == type(_dict)
        assert (dict_serialized_deserialized[0] == tensor_one).all()
        assert (dict_serialized_deserialized[1] == tensor_two).all()

    def test_range_serde(self):
        _range = range(1, 2, 3)

        range_serialized = serialize(_range, compress=False)

        range_serialized_deserialized = deserialize(range_serialized, compressed=False)

        assert _range == range_serialized_deserialized

    def test_compressed_range_serde(self):
        _range = range(1, 2, 3)

        range_serialized = serialize(_range, compress=True)

        range_serialized_deserialized = deserialize(range_serialized, compressed=True)

        assert _range == range_serialized_deserialized

    def test_list(self):
        # Test with integers
        _list = [1, 2]
        list_serialized = serialize(_list, compress=False)
        list_serialized_deserialized = deserialize(list_serialized, compressed=False)
        assert _list == list_serialized_deserialized

        # Test with strings
        _list = ["hello", "world"]
        list_serialized = serialize(_list, compress=False)
        list_serialized_deserialized = deserialize(list_serialized, compressed=False)
        assert _list == list_serialized_deserialized

        # Test with a complex data structure
        tensor_one = Tensor(numpy.random.random((100, 100)))
        tensor_two = Tensor(numpy.random.random((100, 100)))
        _list = (tensor_one, tensor_two)
        list_serialized = serialize(_list, compress=False)
        list_serialized_deserialized = deserialize(list_serialized, compressed=False)
        # `assert list_serialized_deserialized == _list` does not work, therefore it's split
        # into 3 assertions
        assert type(list_serialized_deserialized) == type(_list)
        assert (list_serialized_deserialized[0] == tensor_one).all()
        assert (list_serialized_deserialized[1] == tensor_two).all()

    def test_compressed_list(self):
        # Test with integers
        _list = [1, 2]
        list_serialized = serialize(_list, compress=True)
        list_serialized_deserialized = deserialize(list_serialized, compressed=True)
        assert _list == list_serialized_deserialized

        # Test with strings
        _list = ["hello", "world"]
        list_serialized = serialize(_list, compress=True)
        list_serialized_deserialized = deserialize(list_serialized, compressed=True)
        assert _list == list_serialized_deserialized

        # Test with a complex data structure
        tensor_one = Tensor(numpy.random.random((100, 100)))
        tensor_two = Tensor(numpy.random.random((100, 100)))
        _list = (tensor_one, tensor_two)
        list_serialized = serialize(_list, compress=True)
        list_serialized_deserialized = deserialize(list_serialized, compressed=True)
        # `assert list_serialized_deserialized == _list` does not work, therefore it's split
        # into 3 assertions
        assert type(list_serialized_deserialized) == type(_list)
        assert (list_serialized_deserialized[0] == tensor_one).all()
        assert (list_serialized_deserialized[1] == tensor_two).all()

    def test_set(self):
        # Test with integers
        _set = set([1, 2])
        set_serialized = serialize(_set, compress=False)
        set_serialized_deserialized = deserialize(set_serialized, compressed=False)
        assert _set == set_serialized_deserialized

        # Test with strings
        _set = set(["hello", "world"])
        set_serialized = serialize(_set, compress=False)
        set_serialized_deserialized = deserialize(set_serialized, compressed=False)
        assert _set == set_serialized_deserialized

        # Test with a complex data structure
        tensor_one = Tensor(numpy.random.random((100, 100)))
        tensor_two = Tensor(numpy.random.random((100, 100)))
        _set = (tensor_one, tensor_two)
        set_serialized = serialize(_set, compress=False)
        set_serialized_deserialized = deserialize(set_serialized, compressed=False)
        # `assert set_serialized_deserialized == _set` does not work, therefore it's split
        # into 3 assertions
        assert type(set_serialized_deserialized) == type(_set)
        assert (set_serialized_deserialized[0] == tensor_one).all()
        assert (set_serialized_deserialized[1] == tensor_two).all()

    def test_compressed_set(self):
        # Test with integers
        _set = set([1, 2])
        set_serialized = serialize(_set, compress=True)
        set_serialized_deserialized = deserialize(set_serialized, compressed=True)
        assert _set == set_serialized_deserialized

        # Test with strings
        _set = set(["hello", "world"])
        set_serialized = serialize(_set, compress=True)
        set_serialized_deserialized = deserialize(set_serialized, compressed=True)
        assert _set == set_serialized_deserialized

        # Test with a complex data structure
        tensor_one = Tensor(numpy.random.random((100, 100)))
        tensor_two = Tensor(numpy.random.random((100, 100)))
        _set = (tensor_one, tensor_two)
        set_serialized = serialize(_set, compress=True)
        set_serialized_deserialized = deserialize(set_serialized, compressed=True)
        # `assert set_serialized_deserialized == _set` does not work, therefore it's split
        # into 3 assertions
        assert type(set_serialized_deserialized) == type(_set)
        assert (set_serialized_deserialized[0] == tensor_one).all()
        assert (set_serialized_deserialized[1] == tensor_two).all()

    def test_slice(self):
        s = slice(0, 100, 2)
        x = numpy.random.rand(100)
        s_serialized = serialize(s, compress=False)
        s_serialized_deserialized = deserialize(s_serialized, compressed=False)

        assert type(s) == type(s_serialized_deserialized)
        assert (x[s] == x[s_serialized_deserialized]).all()

        s = slice(40, 50)
        x = numpy.random.rand(100)
        s_serialized = serialize(s, compress=False)
        s_serialized_deserialized = deserialize(s_serialized, compressed=False)

        assert type(s) == type(s_serialized_deserialized)
        assert (x[s] == x[s_serialized_deserialized]).all()

    def test_compressed_slice(self):
        s = slice(0, 100, 2)
        x = numpy.random.rand(100)
        s_serialized = serialize(s, compress=True)
        s_serialized_deserialized = deserialize(s_serialized, compressed=True)

        assert type(s) == type(s_serialized_deserialized)
        assert (x[s] == x[s_serialized_deserialized]).all()

        s = slice(40, 50)
        x = numpy.random.rand(100)
        s_serialized = serialize(s, compress=True)
        s_serialized_deserialized = deserialize(s_serialized, compressed=True)

        assert type(s) == type(s_serialized_deserialized)
        assert (x[s] == x[s_serialized_deserialized]).all()

    def test_PointerTensor(self):
        t = PointerTensor(id=1000, location=self.alice, owner=self.alice)
        t_serialized = serialize(t, compress=False)
        t_serialized_deserialized = deserialize(t_serialized, compressed=False)

        assert t.id == t_serialized_deserialized.id
        assert t.location == t_serialized_deserialized.location
        assert t.owner == t_serialized_deserialized.owner
        assert t.id_at_location == t_serialized_deserialized.id_at_location
