"""
This file exists to provide one common place for all serialization to occur
regardless of framework. As msgpack only supports basic types and binary formats
every type must be first be converted to one of these types. Thus, we've split our
functionality into three steps. When converting from a PySyft object (or collection
of objects) to an object to be sent over the wire (a message), those three steps
are (in order):

1. Simplify - converts PyTorch objects to simple Python objects (using pickle)
2. Serialize - converts Python objects to binary
3. Compress - compresses the binary (Now we're ready send!)

Inversely, when converting from a message sent over the wire back to a PySyft
object, the three steps are (in order):

1. Decompress - converts compressed binary back to decompressed binary
2. Deserialize - converts from binary to basic python objects
3. Detail - converts some basic python objects back to PyTorch objects (Tensors)

Furthermore, note that there is different simplification/serialization logic
for objects of different types. Thus, instead of using if/else logic, we have
global dictionaries which contain functions and Python types as keys. For
simplification logic, this dictionary is called "simplifiers". The keys
are the types and values are the simplification logic. For example,
simplifiers[tuple] will return the function which knows how to simplify the
tuple type. The same is true for all other simplifier/detailer functions.

By default, we serialize using msgpack and compress using lz4.
"""

from typing import Collection
from typing import Dict
from typing import Tuple
import torch
import msgpack
import lz4
from lz4 import (  # noqa: F401
    frame,
)  # needed as otherwise we will get: module 'lz4' has no attribute 'frame'
import io
import numpy
import zstd

# High Level Public Functions (these are the ones you use)


def serialize(obj: object, compress=True, compressScheme="lz4") -> bin:
    """This is the high level function for serializing any object or
    dictionary/collection of objects."""

    # 1) Simplify
    # simplify difficult-to-serialize objects. See the _simpliy method
    # for details on how this works. The general purpose is to handle
    # types which the fast serializer (msgpack) cannot handle
    simple_objects = _simplify(obj)

    # 2) Serialize
    # serialize into a binary
    binary = msgpack.dumps(simple_objects)

    # 3) Compress
    # optionally compress the binary and return the result
    # prepend a 1-byte header '0' or '1' to the output stream
    # to denote whether output stream is compressed or not
    # if compressed stream length is greater than input stream
    # we output the input stream as it is with header set to '0'
    # otherwise we output the compressed stream with header set to '1'
    # even if compressed flag is set to false by the caller we
    # output the input stream as it is with header set to '0'
    if compress:
        compress_stream = _compress(binary, compressScheme)
        if len(compress_stream) < len(binary):
            return b"\x31" + compress_stream

    return b"\x30" + binary


def deserialize(binary: bin, compressed=True, compressScheme="lz4") -> object:
    """
    This is the high level function for deserializing any object
    or dictionary/collection of objects.
    """
    # check the 1-byte header to see if input stream was compressed or not
    if binary[0] == 48:
        compressed = False

    # remove the 1-byte header from the input stream
    binary = binary[1:]
    # 1)  Decompress
    # If enabled, this functionality decompresses the binary
    if compressed:
        binary = _decompress(binary, compressScheme)

    # 2) Deserialize
    # This function converts the binary into the appropriate python
    # object (or nested dict/collection of python objects)
    simple_objects = msgpack.loads(binary)

    # 3) Detail
    # This function converts typed, simple objects into their more
    # complex (and difficult to serialize) counterparts which the
    # serialization library wasn't natively able to serialize (such
    # as msgpack's inability to serialize torch tensors or ... or
    # python slice objects
    return _detail(simple_objects)


# Chosen Compression Algorithm


def _compress(decompressed_input_bin: bin, compressScheme="lz4") -> bin:
    """
    This function compresses a binary using LZ4

    Args:
        bin: binary to be compressed

    Returns:
        bin: a compressed binary

    """
    if compressScheme == "lz4":
        return lz4.frame.compress(decompressed_input_bin)
    else:
        return zstd.compress(decompressed_input_bin)


def _decompress(compressed_input_bin: bin, compressScheme="lz4") -> bin:
    """
    This function decompresses a binary using LZ4

    Args:
        bin: a compressed binary

    Returns:
        bin: decompressed binary

    """
    if compressScheme == "lz4":
        return lz4.frame.decompress(compressed_input_bin)
    else:
        return zstd.decompress(compressed_input_bin)


# Simplify/Detail Torch Tensors


def _simplify_torch_tensor(tensor: torch.Tensor) -> bin:
    """
    This function converts a torch tensor into a serliaized torch tensor
    using pickle. We choose to use this because PyTorch has a custom and
    very fast PyTorch pickler.

    Args:
        torch.Tensor: an input tensor to be serialized

    Returns:
        bin: serialized binary of torch tensor.
    """

    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    tensor_bin = binary_stream.getvalue()
    return tensor_bin


def _detail_torch_tensor(tensor: bin) -> torch.Tensor:
    """
    This function converts a serialized torch tensor into a torch tensor
    using pickle.

    Args:
        bin: serialized binary of torch tensor

    Returns:
        torch.Tensor: a torch tensor that was serialized
    """

    bin_tensor_stream = io.BytesIO(tensor)
    return torch.load(bin_tensor_stream)


# Simplify/Detail Collections (list, set, tuple, etc.)


def _simplify_collection(my_collection: Collection) -> Collection:
    """
    This function is designed to search a collection for any objects
    which may need to be simplified (i.e., torch tensors). It iterates
    through each object in the collection and calls _simplify on it. Finally,
    it returns the output collection as the same type as the input collection
    so that the consuming serialization step knows the correct type info. The
    reverse function to this function is _detail_collection, which undoes
    the functionality of this function.

    Args:
        Collection: a collection of python objects

    Returns:
        Collection: a collection of the same type as the input of simplified
            objects.

    """

    # Step 0: get collection type for later use and itialize empty list
    my_type = type(my_collection)
    pieces = list()

    # Step 1: serialize each part of the collection
    for part in my_collection:
        pieces.append(_simplify(part))

    # Step 2: convert back to original type and return serialization
    if my_type == set:
        return pieces
    return my_type(pieces)


def _detail_collection_list(my_collection: Collection) -> Collection:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a collection of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        Collection: a collection of simple python objects (including binary).

    Returns:
        Collection: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        try:
            pieces.append(_detail(part).decode("utf-8"))  # transform bytes back to string
        except AttributeError:
            pieces.append(_detail(part))

    return pieces


def _detail_collection_set(my_collection: Collection) -> Collection:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a collection of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        Collection: a collection of simple python objects (including binary).

    Returns:
        Collection: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        try:
            pieces.append(_detail(part).decode("utf-8"))  # transform bytes back to string
        except AttributeError:
            pieces.append(_detail(part))
    return set(pieces)


def _detail_collection_tuple(my_tuple: Tuple) -> Tuple:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a tuple of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.
    This is only applicable to tuples. They need special handling because
    `msgpack` is encoding a tuple as a list.

    Args:
        tuple: a collection of simple python objects (including binary).

    Returns:
        tuple: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_tuple:
        pieces.append(_detail(part))

    return tuple(pieces)


# Dictionaries


def _simplify_dictionary(my_dict: Dict) -> Dict:
    pieces = {}
    # for dictionaries we want to simplify both the key and the value
    for key, value in my_dict.items():
        pieces[_simplify(key)] = _simplify(value)

    return pieces


def _detail_dictionary(my_dict: Dict) -> Dict:
    pieces = {}
    # for dictionaries we want to detail both the key and the value
    for key, value in my_dict.items():

        try:
            detailed_key = _detail(key).decode("utf-8")
        except AttributeError:
            detailed_key = _detail(key)

        try:
            detailed_value = _detail(value).decode("utf-8")
        except AttributeError:
            detailed_value = _detail(value)

        pieces[detailed_key] = detailed_value

    return pieces


# Range


def _simplify_range(my_range: range) -> Tuple[int, int, int]:
    """
    This function extracts the start, stop and step from the range.

    Args:
        range: a range object

    Returns:
        list: a list defining the range parameters [start, stop, step]

    Usage:

        range_parameters = _simplify_range(range(1, 3, 4))

        assert range_parameters == [1, 3, 4]

    """

    return (my_range.start, my_range.stop, my_range.step)


def _detail_range(my_range_params: Tuple[int, int, int]) -> range:
    """
    This function extracts the start, stop and step from a tuple.

    Args:
        list: a list defining the range parameters [start, stop, step]

    Returns:
        range: a range object

    Usage:
        new_range = _detail_range([1, 3, 4])

        assert new_range == range(1, 3, 4)

    """

    return range(my_range_params[0], my_range_params[1], my_range_params[2])


#   numpy array


def _simplify_ndarray(my_array: numpy.ndarray) -> Tuple[bin, Tuple, str]:
    """
    This function gets the byte representation of the array
        and stores the dtype and shape for reconstruction

    Args:
        numpy.ndarray: a numpy array

    Returns:
        list: a list holding the byte representation, shape and dtype of the array

    Usage:

        arr_representation = _simplify_ndarray(numpy.random.random([1000, 1000])))

    """
    arr_bytes = my_array.tobytes()
    arr_shape = my_array.shape
    arr_dtype = my_array.dtype.name

    return (arr_bytes, arr_shape, arr_dtype)


def _detail_ndarray(arr_representation: Tuple[bin, Tuple, str]) -> numpy.ndarray:
    """
    This function reconstruct a numpy array from it's byte data, the shape and the dtype
        by first loading the byte data with the appropiate dtype and then reshaping it into the
        original shape

    Args:
        ist: a list holding the byte representation, shape and dtype of the array

    Returns:
        numpy.ndarray: a numpy array

    Usage:
        arr = _detail_ndarray(arr_representation)

    """
    res = numpy.frombuffer(arr_representation[0], dtype=arr_representation[2]).reshape(
        arr_representation[1]
    )

    assert type(res) == numpy.ndarray

    return res


#   slice


def _simplify_slice(my_slice: slice) -> Tuple[int, int, int]:
    """
    This function creates a list that represents a slice.

    Args:
        slice: a python slice

    Returns:
        list: a list holding the start, stop and step values

    Usage:

        slice_representation = _simplify_slice(slice(1,2,3))

    """
    return (my_slice.start, my_slice.stop, my_slice.step)


def _detail_slice(my_slice: Tuple[int, int, int]) -> slice:
    """
    This function extracts the start, stop and step from a list.

    Args:
        list: a list defining the slice parameters [start, stop, step]

    Returns:
        range: a range object

    Usage:
        new_range = _detail_range([1, 3, 4])

        assert new_range == range(1, 3, 4)

    """

    return slice(my_slice[0], my_slice[1], my_slice[2])


# ellipsis


def _simplify_ellipsis(e: Ellipsis) -> bytes:
    return b""


def _detail_ellipsis(ellipsis: bytes) -> Ellipsis:
    return ...

    # High Level Simplification Router


def _simplify(obj: object) -> object:
    """
    This function takes an object as input and returns a simple
    Python object which is supported by the chosen serialization
    method (such as JSON or msgpack). The reason we have this function
    is that some objects are either NOT supported by high level (fast)
    serializers OR the high level serializers don't support the fastest
    form of serialization. For example, PyTorch tensors have custom pickle
    functionality thus its better to pre-serialize PyTorch tensors using
    pickle and then serialize the binary in with the rest of the message
    being sent.

    Args:
        obj: an object which may need to be simplified

    Returns:
        obj: an simple Python object which msgpack can serialize

    Raises:
        ValueError: if `move_this` or `in_front_of_that` are not both single ASCII
        characters.

    """

    try:
        # check to see if there is a simplifier
        # for this type. If there is, run return
        # the simplified object
        current_type = type(obj)
        return [simplifiers[current_type][0], simplifiers[current_type][1](obj)]

    except KeyError:

        # if there is not a simplifier for this
        # object, then the object is already a
        # simple python object and we can just
        # return it
        return obj


simplifiers = {
    torch.Tensor: [0, _simplify_torch_tensor],
    tuple: [1, _simplify_collection],
    list: [2, _simplify_collection],
    set: [3, _simplify_collection],
    dict: [4, _simplify_dictionary],
    range: [5, _simplify_range],
    numpy.ndarray: [6, _simplify_ndarray],
    slice: [7, _simplify_slice],
    type(Ellipsis): [8, _simplify_ellipsis],
}


def _detail(obj: object) -> object:
    """
    This function reverses the functionality of _simplify. Where applicable,
    it converts simple objects into more complex objects such as converting
    binary objects into torch tensors. Read _simplify for more information on
    why _simplify and _detail are needed.

    Args:
        obj: a simple Python object which msgpack deserialized

    Returns:
        obj: a more complex Python object which msgpack would have had trouble
            deserializing directly.

    """
    if type(obj) == list:
        return detailers[obj[0]](obj[1])
    else:
        return obj


detailers = [
    _detail_torch_tensor,
    _detail_collection_tuple,
    _detail_collection_list,
    _detail_collection_set,
    _detail_dictionary,
    _detail_range,
    _detail_ndarray,
    _detail_slice,
    _detail_ellipsis,
]
