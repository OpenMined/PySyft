"""
This file tests the ability for serde.py to convert complex types into
simple python types which are serializable by standard serialization tools.
For more on how/why this works, see serde.py directly.
"""
import msgpack as msgpack_lib
import numpy
import pytest
import torch
from torch import Tensor

import syft
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.serde import compression
from syft.serde import msgpack
from syft.serde import serde
from syft.serde.msgpack import native_serde
from syft.serde.msgpack import torch_serde
from syft.workers.virtual import VirtualWorker

from syft.exceptions import CompressionNotFoundException


def test_tuple_simplify(workers):
    """This tests our ability to simplify tuple types.

    This test is pretty simple since tuples just serialize to
    themselves, with a tuple wrapper with the correct ID (1)
    for tuples so that the detailer knows how to interpret it."""

    me = workers["me"]
    input = ("hello", "world")
    tuple_detail_code = msgpack.proto_type_info(tuple).code
    str_detail_code = msgpack.proto_type_info(str).code
    target = (tuple_detail_code, ((str_detail_code, (b"hello",)), (str_detail_code, (b"world",))))
    assert msgpack.serde._simplify(me, input) == target


def test_list_simplify(workers):
    """This tests our ability to simplify list types.

    This test is pretty simple since lists just serialize to
    themselves, with a tuple wrapper with the correct ID (2)
    for lists so that the detailer knows how to interpret it."""

    me = workers["me"]
    input = ["hello", "world"]
    list_detail_code = msgpack.proto_type_info(list).code
    str_detail_code = msgpack.proto_type_info(str).code
    target = (list_detail_code, ((str_detail_code, (b"hello",)), (str_detail_code, (b"world",))))
    assert msgpack.serde._simplify(me, input) == target


def test_set_simplify(workers):
    """This tests our ability to simplify set objects.

    This test is pretty simple since sets just serialize to
    lists, with a tuple wrapper with the correct ID (3)
    for sets so that the detailer knows how to interpret it."""

    me = workers["me"]
    input = set(["hello", "world"])
    set_detail_code = msgpack.proto_type_info(set).code
    str_detail_code = msgpack.proto_type_info(str).code
    target = (set_detail_code, ((str_detail_code, (b"hello",)), (str_detail_code, (b"world",))))
    assert msgpack.serde._simplify(me, input)[0] == target[0]
    assert set(msgpack.serde._simplify(me, input)[1]) == set(target[1])


def test_float_simplify(workers):
    """This tests our ability to simplify float objects.

    This test is pretty simple since floats just serialize to
    themselves, with no tuple/id necessary."""

    me = workers["me"]
    input = 5.6
    target = 5.6
    assert msgpack.serde._simplify(me, input) == target


def test_int_simplify(workers):
    """This tests our ability to simplify int objects.

    This test is pretty simple since ints just serialize to
    themselves, with no tuple/id necessary."""

    me = workers["me"]
    input = 5
    target = 5
    assert msgpack.serde._simplify(me, input) == target


def test_string_simplify(workers):
    """This tests our ability to simplify string objects.

    This test is pretty simple since strings just serialize to
    themselves, with no tuple/id necessary."""

    me = workers["me"]
    input = "hello"
    target = (msgpack.proto_type_info(str).code, (b"hello",))
    assert msgpack.serde._simplify(me, input) == target


def test_dict_simplify(workers):
    """This tests our ability to simplify dict objects.

    This test is pretty simple since dicts just serialize to
    themselves, with a tuple wrapper with the correct ID
    for dicts so that the detailer knows how to interpret it."""

    me = workers["me"]
    input = {"hello": "world"}
    detail_dict_code = msgpack.proto_type_info(dict).code
    detail_str_code = msgpack.proto_type_info(str).code
    target = (detail_dict_code, (((detail_str_code, (b"hello",)), (detail_str_code, (b"world",))),))
    assert msgpack.serde._simplify(me, input) == target


def test_range_simplify(workers):
    """This tests our ability to simplify range objects.

    This test is pretty simple since range objs just serialize to
    themselves, with a tuple wrapper with the correct ID (5)
    for dicts so that the detailer knows how to interpret it."""

    me = workers["me"]
    input = range(1, 3, 4)
    target = (msgpack.proto_type_info(range).code, (1, 3, 4))
    assert msgpack.serde._simplify(me, input) == target


def test_torch_tensor_simplify(workers):
    """This tests our ability to simplify torch.Tensor objects
    using "torch" serialization strategy.

    At the time of writing, tensors simplify to a tuple where the
    first value in the tuple is the tensor's ID and the second
    value is a serialized version of the Tensor (serialized
    by PyTorch's torch.save method)
    """

    me = workers["me"]

    # create a tensor
    input = Tensor(numpy.random.random((100, 100)))

    # simplify the tnesor
    output = msgpack.serde._simplify(me, input)

    # make sure outer type is correct
    assert type(output) == tuple

    # make sure the object type ID is correct
    # (0 for torch.Tensor)
    assert msgpack.serde.detailers[output[0]] == torch_serde._detail_torch_tensor

    # make sure inner type is correct
    assert type(output[1]) == tuple

    # make sure ID is correctly encoded
    assert output[1][0] == input.id

    # make sure tensor data type is correct
    assert type(output[1][1]) == bytes


def test_torch_tensor_simplify_generic(workers):
    """This tests our ability to simplify torch.Tensor objects
    using "all" serialization strategy
    """

    worker = VirtualWorker(None, id="non-torch")

    # create a tensor
    input = Tensor(numpy.random.random((3, 3, 3)))

    # simplify the tensor
    output = msgpack.serde._simplify(worker, input)

    # make sure outer type is correct
    assert type(output) == tuple

    # make sure the object type ID is correct
    # (0 for torch.Tensor)
    assert msgpack.serde.detailers[output[0]] == torch_serde._detail_torch_tensor

    # make sure inner type is correct
    assert type(output[1]) == tuple

    # make sure ID is correctly encoded
    assert output[1][0] == input.id

    # make sure tensor data type is correct
    assert type(output[1][1]) == tuple
    assert type(output[1][1][1]) == tuple

    # make sure tensor data matches
    assert output[1][1][1][0][1] == input.size()
    assert output[1][1][1][2][1] == tuple(input.flatten().tolist())


def test_torch_tensor_serde_generic(workers):
    """This tests our ability to ser-de torch.Tensor objects
    using "all" serialization strategy
    """

    worker = VirtualWorker(None, id="non-torch")

    # create a tensor
    input = Tensor(numpy.random.random((100, 100)))

    # ser-de the tensor
    output = msgpack.serde._simplify(worker, input)
    detailed = msgpack.serde._detail(worker, output)

    # check tensor contents
    assert input.size() == detailed.size()
    assert input.dtype == detailed.dtype
    assert (input == detailed).all()


def test_tensor_gradient_serde():
    # create a tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True)

    # create gradient on tensor
    x.sum().backward(torch.tensor(1.0))

    # save gradient
    orig_grad = x.grad

    # serialize
    blob = syft.serde.serialize(x)

    # deserialize
    t = syft.serde.deserialize(blob)

    # check that gradient was properly serde
    assert (t.grad == orig_grad).all()


def test_ndarray_simplify(workers):
    """This tests our ability to simplify numpy.array objects

    At the time of writing, arrays simplify to an object inside
    of a tuple which specifies the ID for the np.array type (6) so
    that the detailer knows to turn the simplifed form to a np.array
    """

    me = workers["me"]
    input = numpy.random.random((100, 100))
    output = msgpack.serde._simplify(me, input)

    # make sure simplified type ID is correct
    assert msgpack.serde.detailers[output[0]] == native_serde._detail_ndarray

    # make sure serialized form is correct
    assert type(output[1][0]) == bytes
    assert output[1][1] == msgpack.serde._simplify(me, input.shape)
    assert output[1][2] == msgpack.serde._simplify(me, input.dtype.name)


def test_numpy_number_simplify(workers):
    """This tests our ability to simplify numpy.float objects

    At the time of writing, numpy number simplify to an object inside
    of a tuple where the first value is a byte representation of the number
    and the second value is the dtype
    """
    me = workers["me"]

    input = numpy.float32(2.0)
    output = msgpack.serde._simplify(me, input)

    # make sure simplified type ID is correct
    assert msgpack.serde.detailers[output[0]] == native_serde._detail_numpy_number

    # make sure serialized form is correct
    assert type(output[1][0]) == bytes
    assert output[1][1] == msgpack.serde._simplify(me, input.dtype.name)


def test_ellipsis_simplify(workers):
    """Make sure ellipsis simplifies correctly."""
    me = workers["me"]

    assert (
        msgpack.serde.detailers[msgpack.serde._simplify(me, Ellipsis)[0]]
        == native_serde._detail_ellipsis
    )

    # the simplified ellipsis (empty object)
    assert msgpack.serde._simplify(me, Ellipsis)[1] == (b"",)


def test_torch_device_simplify(workers):
    """Test the simplification of torch.device"""

    me = workers["me"]
    device = torch.device("cpu")

    assert (
        msgpack.serde.detailers[msgpack.serde._simplify(me, device)[0]]
        == torch_serde._detail_torch_device
    )

    # the simplified torch.device
    assert msgpack.serde._simplify(me, device)[1][0] == msgpack.serde._simplify(me, "cpu")


def test_torch_dtype_simplify(workers):
    """Test the simplification of torch.dtype"""

    me = workers["me"]
    dtype = torch.int32

    assert (
        msgpack.serde.detailers[msgpack.serde._simplify(me, dtype)[0]]
        == torch_serde._detail_torch_dtype
    )

    # the simplified torch.dtype
    assert msgpack.serde._simplify(me, dtype)[1] == "int32"


def test_pointer_tensor_simplify(workers):
    """Test the simplification of PointerTensor"""

    alice, me = workers["alice"], workers["me"]

    input_tensor = PointerTensor(id=1000, location=alice, owner=alice)

    output = msgpack.serde._simplify(me, input_tensor)

    assert output[1][0] == input_tensor.id
    assert output[1][1] == input_tensor.id_at_location
    assert output[1][2] == msgpack.serde._simplify(me, input_tensor.owner.id)


@pytest.mark.parametrize("compress", [True, False])
def test_torch_Tensor(compress):
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    t = Tensor(numpy.random.random((100, 100)))
    t_serialized = syft.serde.serialize(t)
    t_serialized_deserialized = syft.serde.deserialize(t_serialized)
    assert (t == t_serialized_deserialized).all()


@pytest.mark.parametrize("compress", [True, False])
def test_torch_Tensor_convenience(compress):
    """This test evaluates torch.Tensor.serialize()

    As opposed to using syft.serde.serialize(), torch objects
    have a convenience function which lets you call .serialize()
    directly on the tensor itself. This tests to makes sure it
    works correctly."""
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    t = Tensor(numpy.random.random((100, 100)))
    t_serialized = t.serialize()
    t_serialized_deserialized = syft.serde.deserialize(t_serialized)
    assert (t == t_serialized_deserialized).all()


@pytest.mark.parametrize("compress", [True, False])
def test_tuple(compress):
    # Test with a simple datatype
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    tuple = (1, 2)
    tuple_serialized = syft.serde.serialize(tuple)
    tuple_serialized_deserialized = syft.serde.deserialize(tuple_serialized)
    assert tuple == tuple_serialized_deserialized

    # Test with a complex data structure
    tensor_one = Tensor(numpy.random.random((100, 100)))
    tensor_two = Tensor(numpy.random.random((100, 100)))
    tuple = (tensor_one, tensor_two)
    tuple_serialized = syft.serde.serialize(tuple)
    tuple_serialized_deserialized = syft.serde.deserialize(tuple_serialized)
    # `assert tuple_serialized_deserialized == tuple` does not work, therefore it's split
    # into 3 assertions
    assert type(tuple_serialized_deserialized) == type(tuple)
    assert (tuple_serialized_deserialized[0] == tensor_one).all()
    assert (tuple_serialized_deserialized[1] == tensor_two).all()


@pytest.mark.parametrize("compress", [True, False])
def test_bytearray(compress):
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    bytearr = bytearray("This is a teststring", "utf-8")
    bytearr_serialized = syft.serde.serialize(bytearr)
    bytearr_serialized_desirialized = syft.serde.deserialize(bytearr_serialized)
    assert bytearr == bytearr_serialized_desirialized

    bytearr = bytearray(numpy.random.random((100, 100)))
    bytearr_serialized = syft.serde.serialize(bytearr)
    bytearr_serialized_desirialized = syft.serde.deserialize(bytearr_serialized)
    assert bytearr == bytearr_serialized_desirialized


@pytest.mark.parametrize("compress", [True, False])
def test_ndarray_serde(compress):
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression
    arr = numpy.random.random((100, 100))
    arr_serialized = syft.serde.serialize(arr)

    arr_serialized_deserialized = syft.serde.deserialize(arr_serialized)

    assert numpy.array_equal(arr, arr_serialized_deserialized)


@pytest.mark.parametrize(
    "compress_scheme", [compression.LZ4, compression.ZLIB, compression.NO_COMPRESSION]
)
def test_compress_decompress(compress_scheme):
    if compress_scheme == compression.LZ4:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    elif compress_scheme == compression.ZLIB:
        compression._apply_compress_scheme = compression.apply_zlib_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    original = msgpack_lib.dumps([1, 2, 3])
    compressed = compression._compress(original)
    decompressed = compression._decompress(compressed)
    assert type(compressed) == bytes
    assert original == decompressed


@pytest.mark.parametrize(
    "compress_scheme", [compression.LZ4, compression.ZLIB, compression.NO_COMPRESSION]
)
def test_compressed_serde(compress_scheme):
    if compress_scheme == compression.LZ4:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    elif compress_scheme == compression.ZLIB:
        compression._apply_compress_scheme = compression.apply_zlib_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    # using numpy.ones because numpy.random.random is not compressed.
    arr = numpy.ones((100, 100))

    arr_serialized = syft.serde.serialize(arr)

    arr_serialized_deserialized = syft.serde.deserialize(arr_serialized)
    assert numpy.array_equal(arr, arr_serialized_deserialized)


@pytest.mark.parametrize("compress", [True, False])
def test_dict(compress):
    # Test with integers
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression
    _dict = {1: 1, 2: 2, 3: 3}
    dict_serialized = syft.serde.serialize(_dict)
    dict_serialized_deserialized = syft.serde.deserialize(dict_serialized)
    assert _dict == dict_serialized_deserialized

    # Test with strings
    _dict = {"one": 1, "two": 2, "three": 3}
    dict_serialized = syft.serde.serialize(_dict)
    dict_serialized_deserialized = syft.serde.deserialize(dict_serialized)
    assert _dict == dict_serialized_deserialized

    # Test with a complex data structure
    tensor_one = Tensor(numpy.random.random((100, 100)))
    tensor_two = Tensor(numpy.random.random((100, 100)))
    _dict = {0: tensor_one, 1: tensor_two}
    dict_serialized = syft.serde.serialize(_dict)
    dict_serialized_deserialized = syft.serde.deserialize(dict_serialized)
    # `assert dict_serialized_deserialized == _dict` does not work, therefore it's split
    # into 3 assertions
    assert type(dict_serialized_deserialized) == type(_dict)
    assert (dict_serialized_deserialized[0] == tensor_one).all()
    assert (dict_serialized_deserialized[1] == tensor_two).all()


@pytest.mark.parametrize("compress", [True, False])
def test_range_serde(compress):
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    _range = range(1, 2, 3)

    range_serialized = syft.serde.serialize(_range)
    range_serialized_deserialized = syft.serde.deserialize(range_serialized)

    assert _range == range_serialized_deserialized


@pytest.mark.parametrize("compress", [True, False])
def test_list(compress):
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    # Test with integers
    _list = [1, 2]
    list_serialized = syft.serde.serialize(_list)
    list_serialized_deserialized = syft.serde.deserialize(list_serialized)
    assert _list == list_serialized_deserialized

    # Test with strings
    _list = ["hello", "world"]
    list_serialized = syft.serde.serialize(_list)
    list_serialized_deserialized = syft.serde.deserialize(list_serialized)
    assert _list == list_serialized_deserialized

    # Test with a complex data structure
    tensor_one = Tensor(numpy.ones((100, 100)))
    tensor_two = Tensor(numpy.ones((100, 100)) * 2)
    _list = (tensor_one, tensor_two)

    list_serialized = syft.serde.serialize(_list)
    if compress:
        assert list_serialized[0] == compression.LZ4
    else:
        assert list_serialized[0] == compression.NO_COMPRESSION

    list_serialized_deserialized = syft.serde.deserialize(list_serialized)
    # `assert list_serialized_deserialized == _list` does not work, therefore it's split
    # into 3 assertions
    assert type(list_serialized_deserialized) == type(_list)
    assert (list_serialized_deserialized[0] == tensor_one).all()
    assert (list_serialized_deserialized[1] == tensor_two).all()


@pytest.mark.parametrize("compress", [True, False])
def test_set(compress):
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    # Test with integers
    _set = set([1, 2])
    set_serialized = syft.serde.serialize(_set)

    set_serialized_deserialized = syft.serde.deserialize(set_serialized)
    assert _set == set_serialized_deserialized

    # Test with strings
    _set = set(["hello", "world"])
    set_serialized = syft.serde.serialize(_set)
    set_serialized_deserialized = syft.serde.deserialize(set_serialized)
    assert _set == set_serialized_deserialized

    # Test with a complex data structure
    tensor_one = Tensor(numpy.ones((100, 100)))
    tensor_two = Tensor(numpy.ones((100, 100)) * 2)
    _set = (tensor_one, tensor_two)

    set_serialized = syft.serde.serialize(_set)
    if compress:
        assert set_serialized[0] == compression.LZ4
    else:
        assert set_serialized[0] == compression.NO_COMPRESSION

    set_serialized_deserialized = syft.serde.deserialize(set_serialized)
    # `assert set_serialized_deserialized == _set` does not work, therefore it's split
    # into 3 assertions
    assert type(set_serialized_deserialized) == type(_set)
    assert (set_serialized_deserialized[0] == tensor_one).all()
    assert (set_serialized_deserialized[1] == tensor_two).all()


@pytest.mark.parametrize("compress", [True, False])
def test_slice(compress):
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    s = slice(0, 100, 2)
    x = numpy.random.rand(100)
    s_serialized = syft.serde.serialize(s)
    s_serialized_deserialized = syft.serde.deserialize(s_serialized)

    assert type(s) == type(s_serialized_deserialized)
    assert (x[s] == x[s_serialized_deserialized]).all()

    s = slice(40, 50)
    x = numpy.random.rand(100)
    s_serialized = syft.serde.serialize(s)
    s_serialized_deserialized = syft.serde.deserialize(s_serialized)

    assert type(s) == type(s_serialized_deserialized)
    assert (x[s] == x[s_serialized_deserialized]).all()


@pytest.mark.parametrize("compress", [True, False])
def test_float(compress):
    if compress:
        compression._apply_compress_scheme = compression.apply_lz4_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    x = 0.5
    y = 1.5

    x_serialized = syft.serde.serialize(x)
    x_serialized_deserialized = syft.serde.deserialize(x_serialized)

    y_serialized = syft.serde.serialize(y)
    y_serialized_deserialized = syft.serde.deserialize(y_serialized)

    assert x_serialized_deserialized == x
    assert y_serialized_deserialized == y


@pytest.mark.parametrize(
    "compress, compress_scheme",
    [
        (True, compression.LZ4),
        (False, compression.LZ4),
        (True, compression.ZLIB),
        (False, compression.ZLIB),
        (True, compression.NO_COMPRESSION),
        (False, compression.NO_COMPRESSION),
    ],
)
def test_hooked_tensor(compress, compress_scheme):
    if compress:
        if compress_scheme == compression.LZ4:
            compression._apply_compress_scheme = compression.apply_lz4_compression
        elif compress_scheme == compression.ZLIB:
            compression._apply_compress_scheme = compression.apply_zlib_compression
        else:
            compression._apply_compress_scheme = compression.apply_no_compression
    else:
        compression._apply_compress_scheme = compression.apply_no_compression

    t = Tensor(numpy.ones((100, 100)))
    t_serialized = syft.serde.serialize(t)
    assert (
        t_serialized[0] == compress_scheme
        if compress
        else t_serialized[0] == compression.NO_COMPRESSION
    )
    t_serialized_deserialized = syft.serde.deserialize(t_serialized)
    assert (t == t_serialized_deserialized).all()


def test_pointer_tensor(hook, workers):
    compression._apply_compress_scheme = compression.apply_no_compression
    t = PointerTensor(
        id=1000, location=workers["alice"], owner=workers["alice"], id_at_location=12345
    )
    t_serialized = syft.serde.serialize(t)
    t_serialized_deserialized = syft.serde.deserialize(t_serialized)
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


@pytest.mark.parametrize(
    "tensor",
    [
        (torch.tensor(numpy.ones((10, 10)), requires_grad=False)),
        (torch.tensor([[0.25, 1.5], [0.15, 0.25], [1.25, 0.5]], requires_grad=True)),
        (torch.randint(low=0, high=10, size=[3, 7], requires_grad=False)),
    ],
)
def test_numpy_tensor_serde(tensor):
    compression._apply_compress_scheme = compression.apply_lz4_compression

    serde._serialize_tensor = syft.serde.msgpack.torch_serde.numpy_tensor_serializer
    serde._deserialize_tensor = syft.serde.msgpack.torch_serde.numpy_tensor_deserializer

    tensor_serialized = syft.serde.serialize(tensor)
    assert tensor_serialized[0] != compression.NO_COMPRESSION
    tensor_deserialized = syft.serde.deserialize(tensor_serialized)

    # Back to Pytorch serializer
    serde._serialize_tensor = syft.serde.msgpack.torch_serde.torch_tensor_serializer
    serde._deserialize_tensor = syft.serde.msgpack.torch_serde.torch_tensor_deserializer

    assert torch.eq(tensor_deserialized, tensor).all()


@pytest.mark.parametrize("compress", [True, False])
def test_additive_sharing_tensor_serde(compress, workers):
    alice, bob, james, me = workers["alice"], workers["bob"], workers["james"], workers["me"]

    x = torch.tensor([[3.1, 4.3]]).fix_prec().share(alice, bob, crypto_provider=james)

    additive_sharing_tensor = x.child.child
    data = AdditiveSharingTensor.simplify(me, additive_sharing_tensor)
    additive_sharing_tensor_reconstructed = AdditiveSharingTensor.detail(me, data)

    assert additive_sharing_tensor_reconstructed.field == additive_sharing_tensor.field

    assert (
        additive_sharing_tensor_reconstructed.child.keys() == additive_sharing_tensor.child.keys()
    )


@pytest.mark.parametrize("compress", [True, False])
def test_fixed_precision_tensor_serde(compress, workers):
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    x = (
        torch.tensor([[3.1, 4.3]])
        .fix_prec(base=12, precision_fractional=5)
        .share(alice, bob, crypto_provider=james)
    )

    serialized_x = syft.serde.serialize(x)
    deserialized_x = syft.serde.deserialize(serialized_x)

    assert x.id == deserialized_x.child.id
    assert x.child.field == deserialized_x.child.field
    assert x.child.kappa == deserialized_x.child.kappa
    assert x.child.precision_fractional == deserialized_x.child.precision_fractional
    assert x.child.base == deserialized_x.child.base


def test_serde_object_wrapper_int():
    obj = 4
    obj_wrapper = ObjectWrapper(obj, id=100)
    msg = syft.serde.serialize(obj_wrapper)

    obj_wrapper_received = syft.serde.deserialize(msg)

    assert obj_wrapper.obj == obj_wrapper_received.obj
    assert obj_wrapper.id == obj_wrapper_received.id


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_serialize_and_deserialize_torch_scriptmodule():  # pragma: no cover
    @torch.jit.script
    def foo(x):
        return x + 2

    bin_message = torch_serde._simplify_script_module(foo)
    foo_loaded = torch_serde._detail_script_module(None, bin_message)

    assert foo.code == foo_loaded.code


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_torch_jit_script_module_serde():  # pragma: no cover
    @torch.jit.script
    def foo(x):
        return x + 2

    msg = syft.serde.serialize(foo)
    foo_received = syft.serde.deserialize(msg)

    assert foo.code == foo_received.code


def test_serde_virtual_worker(hook):
    virtual_worker = syft.VirtualWorker(hook=hook, id="deserialized_worker1")
    # Populate worker
    tensor1, tensor2 = torch.tensor([1.0, 2.0]), torch.tensor([0.0])
    ptr1, ptr2 = tensor1.send(virtual_worker), tensor2.send(virtual_worker)

    serialized_worker = syft.serde.serialize(virtual_worker, force_full_simplification=False)
    deserialized_worker = syft.serde.deserialize(serialized_worker)

    assert virtual_worker.id == deserialized_worker.id


def test_full_serde_virtual_worker(hook):
    virtual_worker = syft.VirtualWorker(hook=hook, id="deserialized_worker2")
    # Populate worker
    tensor1, tensor2 = torch.tensor([1.0, 2.0]), torch.tensor([0.0])
    ptr1, ptr2 = tensor1.send(virtual_worker), tensor2.send(virtual_worker)

    serialized_worker = syft.serde.serialize(virtual_worker, force_full_simplification=True)

    deserialized_worker = syft.serde.deserialize(serialized_worker)

    assert virtual_worker.id == deserialized_worker.id
    assert virtual_worker.auto_add == deserialized_worker.auto_add
    assert len(deserialized_worker.object_store._tensors) == 2
    assert tensor1.id in deserialized_worker.object_store._tensors
    assert tensor2.id in deserialized_worker.object_store._tensors


def test_serde_object_wrapper_traced_module():

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]])

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(2, 3)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            return x

    obj = torch.jit.trace(Net(), data)

    obj_wrapper = ObjectWrapper(obj, id=200)
    msg = syft.serde.serialize(obj_wrapper)

    obj_wrapper_received = syft.serde.deserialize(msg)

    pred_before = obj(data)

    pred_after = obj_wrapper_received.obj(data)

    assert (pred_before == pred_after).all()
    assert obj_wrapper.id == obj_wrapper_received.id


def test_no_simplifier_found(workers):
    """Test that types that can not be simplified are cached."""
    me = workers["me"]
    # Clean cache.
    msgpack.serde.no_simplifiers_found = set()
    x = 1.3
    assert type(x) not in msgpack.serde.no_simplifiers_found
    _ = msgpack.serde._simplify(me, x)
    assert type(x) in msgpack.serde.no_simplifiers_found
