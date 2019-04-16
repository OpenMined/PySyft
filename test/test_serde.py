"""
This file tests the ability for serde.py to convert complex types into
simple python types which are serializable by standard serialization tools.
For more on how/why this works, see serde.py directly.
"""
import warnings

from syft.serde import (
    _simplify,
    apply_lz4_compression,
    apply_no_compression,
    apply_zstd_compression,
)
from syft.serde import serialize
from syft.serde import deserialize
from syft.serde import _compress
from syft.serde import _decompress
from syft.serde import LZ4
from syft.serde import NO_COMPRESSION
from syft.serde import ZSTD


import syft
from syft.exceptions import CompressionNotFoundException
from syft.frameworks.torch.tensors.interpreters import PointerTensor

import msgpack
import numpy
import pytest
import torch
from torch import Tensor


def test_tuple_simplify():
    """This tests our ability to simplify tuple types.

    This test is pretty simple since tuples just serialize to
    themselves, with a tuple wrapper with the correct ID (1)
    for tuples so that the detailer knows how to interpret it."""

    input = ("hello", "world")
    target = (2, ((18, (b"hello",)), (18, (b"world",))))
    assert _simplify(input) == target


def test_list_simplify():
    """This tests our ability to simplify list types.

    This test is pretty simple since lists just serialize to
    themselves, with a tuple wrapper with the correct ID (2)
    for lists so that the detailer knows how to interpret it."""

    input = ["hello", "world"]
    target = (3, [(18, (b"hello",)), (18, (b"world",))])
    assert _simplify(input) == target


def test_set_simplify():
    """This tests our ability to simplify set objects.

    This test is pretty simple since sets just serialize to
    lists, with a tuple wrapper with the correct ID (3)
    for sets so that the detailer knows how to interpret it."""

    input = set(["hello", "world"])
    target = (4, [(18, (b"hello",)), (18, (b"world",))])
    assert _simplify(input)[0] == target[0]
    assert set(_simplify(input)[1]) == set(target[1])


def test_float_simplify():
    """This tests our ability to simplify float objects.

    This test is pretty simple since floats just serialize to
    themselves, with no tuple/id necessary."""

    input = 5.6
    target = 5.6
    assert _simplify(input) == target


def test_int_simplify():
    """This tests our ability to simplify int objects.

    This test is pretty simple since ints just serialize to
    themselves, with no tuple/id necessary."""

    input = 5
    target = 5
    assert _simplify(input) == target


def test_string_simplify():
    """This tests our ability to simplify string objects.

    This test is pretty simple since strings just serialize to
    themselves, with no tuple/id necessary."""

    input = "hello"
    target = (18, (b"hello",))
    assert _simplify(input) == target


def test_dict_simplify():
    """This tests our ability to simplify dict objects.

    This test is pretty simple since dicts just serialize to
    themselves, with a tuple wrapper with the correct ID (4)
    for dicts so that the detailer knows how to interpret it."""

    input = {"hello": "world"}
    target = (5, [((18, (b"hello",)), (18, (b"world",)))])
    assert _simplify(input) == target


def test_range_simplify():
    """This tests our ability to simplify range objects.

    This test is pretty simple since range objs just serialize to
    themselves, with a tuple wrapper with the correct ID (5)
    for dicts so that the detailer knows how to interpret it."""

    input = range(1, 3, 4)
    target = (6, (1, 3, 4))
    assert _simplify(input) == target


def test_torch_tensor_simplify():
    """This tests our ability to simplify torch.Tensor objects

    At the time of writing, tensors simplify to a tuple where the
    first value in the tuple is the tensor's ID and the second
    value is a serialized version of the Tensor (serialized
    by PyTorch's torch.save method)
    """

    # create a tensor
    input = Tensor(numpy.random.random((100, 100)))

    # simplify the tnesor
    output = _simplify(input)

    # make sure outer type is correct
    assert type(output) == tuple

    # make sure the object type ID is correct
    # (0 for torch.Tensor)
    assert output[0] == 0

    # make sure inner type is correct
    assert type(output[1]) == tuple

    # make sure ID is correctly encoded
    assert output[1][0] == input.id

    # make sure tensor data type is correct
    assert type(output[1][1]) == bytes


def test_ndarray_simplify():
    """This tests our ability to simplify numpy.array objects

    At the time of writing, arrays simplify to an object inside
    of a tuple which specifies the ID for the np.array type (6) so
    that the detailer knows to turn the simplifed form to a np.array
    """

    input = numpy.random.random((100, 100))
    output = _simplify(input)

    # make sure simplified type ID is correct
    assert output[0] == 7

    # make sure serialized form is correct
    assert type(output[1][0]) == bytes
    assert output[1][1] == input.shape
    assert output[1][2] == input.dtype.name


def test_ellipsis_simplify():
    """Make sure ellipsis simplifies correctly."""

    # the id indicating an ellipsis is here
    assert _simplify(Ellipsis)[0] == 9

    # the simplified ellipsis (empty object)
    assert _simplify(Ellipsis)[1] == b""


def test_torch_device_simplify():
    """Test the simplification of torch.device"""
    device = torch.device("cpu")

    # the id indicating an torch.device is here
    assert _simplify(device)[0] == 10

    # the simplified torch.device
    assert _simplify(device)[1] == "cpu"


def test_pointer_tensor_simplify():
    """Test the simplification of PointerTensor"""

    alice = syft.VirtualWorker(syft.torch.hook, id="alice")
    input_tensor = PointerTensor(id=1000, location=alice, owner=alice)

    output = _simplify(input_tensor)

    assert output[1][0] == input_tensor.id
    assert output[1][1] == input_tensor.id_at_location
    assert output[1][2] == input_tensor.owner.id


@pytest.mark.parametrize("compress", [True, False])
def test_torch_Tensor(compress):
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    t = Tensor(numpy.random.random((100, 100)))
    t_serialized = serialize(t)
    t_serialized_deserialized = deserialize(t_serialized)
    assert (t == t_serialized_deserialized).all()


@pytest.mark.parametrize("compress", [True, False])
def test_torch_Tensor_convenience(compress):
    """This test evaluates torch.Tensor.serialize()

    As opposed to using syft.serde.serialize(), torch objects
    have a convenience function which lets you call .serialize()
    directly on the tensor itself. This tests to makes sure it
    works correctly."""
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    t = Tensor(numpy.random.random((100, 100)))
    t_serialized = t.serialize()
    t_serialized_deserialized = deserialize(t_serialized)
    assert (t == t_serialized_deserialized).all()


@pytest.mark.parametrize("compress", [True, False])
def test_tuple(compress):
    # Test with a simple datatype
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    tuple = (1, 2)
    tuple_serialized = serialize(tuple)
    tuple_serialized_deserialized = deserialize(tuple_serialized)
    assert tuple == tuple_serialized_deserialized

    # Test with a complex data structure
    tensor_one = Tensor(numpy.random.random((100, 100)))
    tensor_two = Tensor(numpy.random.random((100, 100)))
    tuple = (tensor_one, tensor_two)
    tuple_serialized = serialize(tuple)
    tuple_serialized_deserialized = deserialize(tuple_serialized)
    # `assert tuple_serialized_deserialized == tuple` does not work, therefore it's split
    # into 3 assertions
    assert type(tuple_serialized_deserialized) == type(tuple)
    assert (tuple_serialized_deserialized[0] == tensor_one).all()
    assert (tuple_serialized_deserialized[1] == tensor_two).all()


@pytest.mark.parametrize("compress", [True, False])
def test_bytearray(compress):
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    bytearr = bytearray("This is a teststring", "utf-8")
    bytearr_serialized = serialize(bytearr)
    bytearr_serialized_desirialized = deserialize(bytearr_serialized)
    assert bytearr == bytearr_serialized_desirialized

    bytearr = bytearray(numpy.random.random((100, 100)))
    bytearr_serialized = serialize(bytearr)
    bytearr_serialized_desirialized = deserialize(bytearr_serialized)
    assert bytearr == bytearr_serialized_desirialized


@pytest.mark.parametrize("compress", [True, False])
def test_ndarray_serde(compress):
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression
    arr = numpy.random.random((100, 100))
    arr_serialized = serialize(arr)

    arr_serialized_deserialized = deserialize(arr_serialized)

    assert numpy.array_equal(arr, arr_serialized_deserialized)


@pytest.mark.parametrize("compress_scheme", [LZ4, ZSTD, NO_COMPRESSION])
def test_compress_decompress(compress_scheme):
    if compress_scheme == LZ4:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    elif compress_scheme == ZSTD:
        syft.serde._apply_compress_scheme = apply_zstd_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    original = msgpack.dumps([1, 2, 3])
    compressed = _compress(original)
    decompressed = _decompress(compressed)
    assert type(compressed) == bytes
    assert original == decompressed


@pytest.mark.parametrize("compress_scheme", [LZ4, ZSTD, NO_COMPRESSION])
def test_compressed_serde(compress_scheme):
    if compress_scheme == LZ4:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    elif compress_scheme == ZSTD:
        syft.serde._apply_compress_scheme = apply_zstd_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    arr = numpy.random.random((100, 100))
    arr_serialized = serialize(arr)

    arr_serialized_deserialized = deserialize(arr_serialized)
    assert numpy.array_equal(arr, arr_serialized_deserialized)


@pytest.mark.parametrize("compress_scheme", [1, 2, 3, 100])
def test_invalid_decompression_scheme(compress_scheme):
    # using numpy.ones because numpy.random.random is not compressed.
    arr = numpy.ones((100, 100))

    def some_other_compression_scheme(decompressed_input):
        # Simulate compression by removing some values
        return decompressed_input[:10], compress_scheme

    syft.serde._apply_compress_scheme = some_other_compression_scheme
    arr_serialized = serialize(arr)
    with pytest.raises(CompressionNotFoundException):
        _ = deserialize(arr_serialized)


@pytest.mark.parametrize("compress", [True, False])
def test_dict(compress):
    # Test with integers
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression
    _dict = {1: 1, 2: 2, 3: 3}
    dict_serialized = serialize(_dict)
    dict_serialized_deserialized = deserialize(dict_serialized)
    assert _dict == dict_serialized_deserialized

    # Test with strings
    _dict = {"one": 1, "two": 2, "three": 3}
    dict_serialized = serialize(_dict)
    dict_serialized_deserialized = deserialize(dict_serialized)
    assert _dict == dict_serialized_deserialized

    # Test with a complex data structure
    tensor_one = Tensor(numpy.random.random((100, 100)))
    tensor_two = Tensor(numpy.random.random((100, 100)))
    _dict = {0: tensor_one, 1: tensor_two}
    dict_serialized = serialize(_dict)
    dict_serialized_deserialized = deserialize(dict_serialized)
    # `assert dict_serialized_deserialized == _dict` does not work, therefore it's split
    # into 3 assertions
    assert type(dict_serialized_deserialized) == type(_dict)
    assert (dict_serialized_deserialized[0] == tensor_one).all()
    assert (dict_serialized_deserialized[1] == tensor_two).all()


@pytest.mark.parametrize("compress", [True, False])
def test_range_serde(compress):
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    _range = range(1, 2, 3)

    range_serialized = serialize(_range)

    range_serialized_deserialized = deserialize(range_serialized)

    assert _range == range_serialized_deserialized


@pytest.mark.parametrize("compress", [True, False])
def test_list(compress):
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    # Test with integers
    _list = [1, 2]
    list_serialized = serialize(_list)
    list_serialized_deserialized = deserialize(list_serialized)
    assert _list == list_serialized_deserialized

    # Test with strings
    _list = ["hello", "world"]
    list_serialized = serialize(_list)
    list_serialized_deserialized = deserialize(list_serialized)
    assert _list == list_serialized_deserialized

    # Test with a complex data structure
    tensor_one = Tensor(numpy.random.random((100, 100)))
    tensor_two = Tensor(numpy.random.random((100, 100)))
    _list = (tensor_one, tensor_two)
    list_serialized = serialize(_list)
    list_serialized_deserialized = deserialize(list_serialized)
    # `assert list_serialized_deserialized == _list` does not work, therefore it's split
    # into 3 assertions
    assert type(list_serialized_deserialized) == type(_list)
    assert (list_serialized_deserialized[0] == tensor_one).all()
    assert (list_serialized_deserialized[1] == tensor_two).all()


@pytest.mark.parametrize("compress", [True, False])
def test_set(compress):
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    # Test with integers
    _set = set([1, 2])
    set_serialized = serialize(_set)
    set_serialized_deserialized = deserialize(set_serialized)
    assert _set == set_serialized_deserialized

    # Test with strings
    _set = set(["hello", "world"])
    set_serialized = serialize(_set)
    set_serialized_deserialized = deserialize(set_serialized)
    assert _set == set_serialized_deserialized

    # Test with a complex data structure
    tensor_one = Tensor(numpy.random.random((100, 100)))
    tensor_two = Tensor(numpy.random.random((100, 100)))
    _set = (tensor_one, tensor_two)
    set_serialized = serialize(_set)
    set_serialized_deserialized = deserialize(set_serialized)
    # `assert set_serialized_deserialized == _set` does not work, therefore it's split
    # into 3 assertions
    assert type(set_serialized_deserialized) == type(_set)
    assert (set_serialized_deserialized[0] == tensor_one).all()
    assert (set_serialized_deserialized[1] == tensor_two).all()


@pytest.mark.parametrize("compress", [True, False])
def test_slice(compress):
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    s = slice(0, 100, 2)
    x = numpy.random.rand(100)
    s_serialized = serialize(s)
    s_serialized_deserialized = deserialize(s_serialized)

    assert type(s) == type(s_serialized_deserialized)
    assert (x[s] == x[s_serialized_deserialized]).all()

    s = slice(40, 50)
    x = numpy.random.rand(100)
    s_serialized = serialize(s)
    s_serialized_deserialized = deserialize(s_serialized)

    assert type(s) == type(s_serialized_deserialized)
    assert (x[s] == x[s_serialized_deserialized]).all()


@pytest.mark.parametrize("compress", [True, False])
def test_float(compress):
    if compress:
        syft.serde._apply_compress_scheme = apply_lz4_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    x = 0.5
    y = 1.5

    x_serialized = serialize(x)
    x_serialized_deserialized = deserialize(x_serialized)

    y_serialized = serialize(y)
    y_serialized_deserialized = deserialize(y_serialized)

    assert x_serialized_deserialized == x
    assert y_serialized_deserialized == y


def test_compressed_float():
    x = 0.5
    y = 1.5

    x_serialized = serialize(x)
    x_serialized_deserialized = deserialize(x_serialized)

    y_serialized = serialize(y)
    y_serialized_deserialized = deserialize(y_serialized)

    assert x_serialized_deserialized == x
    assert y_serialized_deserialized == y


@pytest.mark.parametrize(
    "compress, compress_scheme",
    [
        (True, LZ4),
        (False, LZ4),
        (True, ZSTD),
        (False, ZSTD),
        (True, NO_COMPRESSION),
        (False, NO_COMPRESSION),
    ],
)
def test_hooked_tensor(compress, compress_scheme):
    if compress:
        if compress_scheme == LZ4:
            syft.serde._apply_compress_scheme = apply_lz4_compression
        elif compress_scheme == ZSTD:
            syft.serde._apply_compress_scheme = apply_zstd_compression
        else:
            syft.serde._apply_compress_scheme = apply_no_compression
    else:
        syft.serde._apply_compress_scheme = apply_no_compression

    t = Tensor(numpy.random.random((100, 100)))
    t_serialized = serialize(t)
    t_serialized_deserialized = deserialize(t_serialized)
    assert (t == t_serialized_deserialized).all()


def test_PointerTensor(hook, workers):
    syft.serde._apply_compress_scheme = apply_no_compression
    t = PointerTensor(
        id=1000, location=workers["alice"], owner=workers["alice"], id_at_location=12345
    )
    t_serialized = serialize(t)
    t_serialized_deserialized = deserialize(t_serialized)
    print(f"t.location - {t.location}")
    print(f"t_serialized_deserialized.location - {t_serialized_deserialized.location}")
    assert t.id == t_serialized_deserialized.id
    assert t.location.id == t_serialized_deserialized.location.id
    assert t.id_at_location == t_serialized_deserialized.id_at_location


@pytest.mark.parametrize("id", [1000, "1000"])
def test_pointer_tensor_detail(id):
    alice = syft.VirtualWorker(syft.torch.hook, id=id)
    x = torch.tensor([1, -1, 3, 4])
    x_ptr = x.send(alice)
    x_ptr = 2 * x_ptr
    x_back = x_ptr.get()
    assert (x_back == 2 * x).all()


def test_numpy_tensor_serde():
    syft.serde._serialize_tensor = syft.serde.numpy_tensor_serializer
    syft.serde._deserialize_tensor = syft.serde.numpy_tensor_deserializer

    tensor = torch.tensor(numpy.random.random((10, 10)), requires_grad=False)

    tensor_serialized = serialize(tensor)
    tensor_deserialized = deserialize(tensor_serialized)

    # Back to Pytorch serializer
    syft.serde._serialize_tensor = syft.serde.torch_tensor_serializer
    syft.serde._deserialize_tensor = syft.serde.torch_tensor_deserializer

    assert torch.eq(tensor_deserialized, tensor).all()


@pytest.mark.parametrize("compress", [True, False])
def test_additive_sharing_tensor_serde(compress, workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x = torch.tensor([[3.1, 4.3]]).fix_prec().share(alice, bob, crypto_provider=james)

    additive_sharing_tensor = x.child.child.child
    data = syft.serde._simplify_additive_shared_tensor(additive_sharing_tensor)
    additive_sharing_tensor_reconstructed = syft.serde._detail_additive_shared_tensor(
        syft.hook.local_worker, data
    )

    assert additive_sharing_tensor_reconstructed.field == additive_sharing_tensor.field

    assert (
        additive_sharing_tensor_reconstructed.child.keys() == additive_sharing_tensor.child.keys()
    )
