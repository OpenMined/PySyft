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

By default, the simplification/detail operations expect Torch tensors. If the setup requires other
serialization process, it can override the functions _serialize_tensor and _deserialize_tensor

By default, we serialize using msgpack and compress using lz4.
If different compressions are required, the worker can override the function _apply_compress_scheme
"""
import torch
import msgpack
import lz4
from lz4 import (  # noqa: F401
    frame,
)  # needed as otherwise we will get: module 'lz4' has no attribute 'frame'
import numpy
import zstd

import syft as sy

from syft.federated import TrainConfig

from syft.workers import AbstractWorker  #
from syft.workers import VirtualWorker

from syft.federated import Plan

from syft.exceptions import CompressionNotFoundException
from syft.exceptions import GetNotPermittedError
from syft.exceptions import ResponseSignatureError


from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters import MultiPointerTensor
from syft.frameworks.torch import pointers


from syft.serde.native_serde import (
    _simplify_str,
    _simplify_range,
    _simplify_ellipsis,
    _simplify_slice,
    _detail_str,
    _detail_range,
    _detail_ellipsis,
    _detail_slice,
)

from syft.serde.torch_serde import (
    _detail_torch_tensor,
    _detail_torch_parameter,
    _detail_collection_tuple,
    _detail_collection_list,
    _detail_collection_set,
    _detail_dictionary,
    _detail_ndarray,
    _detail_torch_device,
    force_full_detail,
    _detail_script_module,
    _simplify_torch_tensor,
    _simplify_torch_parameter,
    _simplify_collection,
    _simplify_dictionary,
    _simplify_ndarray,
    _simplify_torch_device,
    _force_fullsimplify,
    _simplify_script_module,
)


# COMPRESSION SCHEME INT CODES
NO_COMPRESSION = 40
LZ4 = 41
ZSTD = 42


# High Level Public Functions (these are the ones you use)
def serialize(
    obj: object,
    simplified: bool = False,
    force_no_compression: bool = False,
    force_no_serialization: bool = False,
    force_full_simplification: bool = False,
) -> bin:
    """This method can serialize any object PySyft needs to send or store.

    This is the high level function for serializing any object or collection
    of objects which PySyft needs to send over the wire. It includes three
    steps, Simplify, Serialize, and Compress as described inline below.

    Args:
        obj (object): the object to be serialized
        simplified (bool): in some cases we want to pass in data which has
            already been simplified - in which case we must skip double
            simplification - which would be bad.... so bad... so... so bad
        force_no_compression (bool): If true, this will override ANY module
            settings and not compress the objects being serialized. The primary
            expected use of this functionality is testing and/or experimentation.
        force_no_serialization (bool): Primarily a testing tool, this will force
            this method to return human-readable Python objects which is very useful
            for testing and debugging (forceably overrides module compression,
            serialization, and the 'force_no_compression' override)). In other words,
            only simplification operations are performed.
        force_full_simplification (bool): Some objects are only partially serialized
            by default. For objects where this is the case, setting this flag to True
            will force the entire object to be serialized. For example, setting this
            flag to True will cause a VirtualWorker to be serialized WITH all of its
            tensors while by default VirtualWorker objects only serialize a small
            amount of metadata.

    Returns:
        binary: the serialized form of the object.

    """
    # 1) Simplify
    # simplify difficult-to-serialize objects. See the _simpliy method
    # for details on how this works. The general purpose is to handle types
    # which the fast serializer cannot handle
    if not simplified:
        if force_full_simplification:
            simple_objects = _force_fullsimplify(obj)
        else:
            simple_objects = _simplify(obj)
    else:
        simple_objects = obj

    # 2) Serialize
    # serialize into a binary
    if force_no_serialization:
        return simple_objects
    else:
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
    if force_no_compression:
        return binary
    else:
        return _compress(binary)


def deserialize(binary: bin, worker: AbstractWorker = None, details=True) -> object:
    """ This method can deserialize any object PySyft needs to send or store.

    This is the high level function for deserializing any object or collection
    of objects which PySyft has sent over the wire or stored. It includes three
    steps, Decompress, Deserialize, and Detail as described inline below.

    Args:
        binary (bin): the serialized object to be deserialized.
        worker (AbstractWorker): the worker which is acquiring the message content,
            for example used to specify the owner of a tensor received(not obvious
            for virtual workers)
        detail (bool): there are some cases where we need to perform the decompression
            and deserialization part, but we don't need to detail all the message.
            This is the case for Plan workers for instance

    Returns:
        object: the deserialized form of the binary input.
    """
    if worker is None:
        worker = sy.torch.hook.local_worker

    # 1) Decompress the binary if needed
    binary = _decompress(binary)

    # 2) Deserialize
    # This function converts the binary into the appropriate python
    # object (or nested dict/collection of python objects)
    simple_objects = msgpack.loads(binary)

    if details:
        # 3) Detail
        # This function converts typed, simple objects into their morefrom typing import Dict
        # complex (and difficult to serialize) counterparts which the
        # serialization library wasn't natively able to serialize (such
        # as msgpack's inability to serialize torch tensors or ... or
        # python slice objects
        return _detail(worker, simple_objects)

    else:
        # sometimes we want to skip detailing (such as in Plan)
        return simple_objects


# Chosen Compression Algorithm


def _apply_compress_scheme(decompressed_input_bin) -> tuple:
    """
    Apply the selected compression scheme.
    By default is used LZ4

    Args:
        decompressed_input_bin: the binary to be compressed
    """
    return apply_lz4_compression(decompressed_input_bin)


def apply_lz4_compression(decompressed_input_bin) -> tuple:
    """
    Apply LZ4 compression to the input

    Args:
        :param decompressed_input_bin: the binary to be compressed
        :return: a tuple (compressed_result, LZ4)
    """
    return lz4.frame.compress(decompressed_input_bin), LZ4


def apply_zstd_compression(decompressed_input_bin) -> tuple:
    """
    Apply ZSTD compression to the input

    Args:
        :param decompressed_input_bin: the binary to be compressed
        :return: a tuple (compressed_result, ZSTD)
    """

    return zstd.compress(decompressed_input_bin), ZSTD


def apply_no_compression(decompressed_input_bin) -> tuple:
    """
    No compression is applied to the input

    Args:
        :param decompressed_input_bin: the binary
        :return: a tuple (the binary, LZ4)
    """

    return decompressed_input_bin, NO_COMPRESSION


def _compress(decompressed_input_bin: bin) -> bin:
    """
    This function compresses a binary using the function _apply_compress_scheme
    if the input has been already compressed in some step, it will return it as it is

    Args:
        decompressed_input_bin (bin): binary to be compressed

    Returns:
        bin: a compressed binary

    """

    compress_stream, compress_scheme = _apply_compress_scheme(decompressed_input_bin)

    if len(compress_stream) < len(decompressed_input_bin):
        return compress_scheme.to_bytes(1, byteorder="big") + compress_stream
    else:
        return NO_COMPRESSION.to_bytes(1, byteorder="big") + decompressed_input_bin


def _decompress(binary: bin) -> bin:
    """
    This function decompresses a binary using the scheme defined in the first byte of the input

    Args:
        binary (bin): a compressed binary

    Returns:
        bin: decompressed binary

    """

    # check the 1-byte header to check the compression scheme used
    compress_scheme = binary[0]

    # remove the 1-byte header from the input stream
    binary = binary[1:]
    # 1)  Decompress or return the original stream
    if compress_scheme == LZ4:
        return lz4.frame.decompress(binary)
    elif compress_scheme == ZSTD:
        return zstd.decompress(binary)
    elif compress_scheme == NO_COMPRESSION:
        return binary
    else:
        raise CompressionNotFoundException(
            "compression scheme not found for" " compression code:" + str(compress_scheme)
        )


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

        result = (simplifiers[current_type][0], simplifiers[current_type][1](obj))

        return result

    except KeyError:

        # if there is not a simplifier for this
        # object, then the object is already a
        # simple python object and we can just
        # return it
        return obj


def _force_full_simplify(obj: object) -> object:
    current_type = type(obj)

    if current_type in forced_full_simplifiers:

        left = forced_full_simplifiers[current_type][0]

        right = forced_full_simplifiers[current_type][1]

        right = right(obj)

        result = (left, right)
    else:
        result = _simplify(obj)

    return result


simplifiers = {
    torch.Tensor: [0, _simplify_torch_tensor],
    torch.nn.Parameter: [1, _simplify_torch_parameter],
    tuple: [2, _simplify_collection],
    list: [3, _simplify_collection],
    set: [4, _simplify_collection],
    dict: [5, _simplify_dictionary],
    range: [6, _simplify_range],
    numpy.ndarray: [7, _simplify_ndarray],
    slice: [8, _simplify_slice],
    type(Ellipsis): [9, _simplify_ellipsis],
    torch.device: [10, _simplify_torch_device],
    pointers.PointerTensor: [11, sy.PointerTensor.simplify],
    LoggingTensor: [12, sy.LoggingTensor.simplify],
    AdditiveSharingTensor: [13, sy.AdditiveSharingTensor.simplify],
    MultiPointerTensor: [14, sy.MultiPointerTensor.simplify],
    Plan: [15, sy.Plan.simplify],
    VirtualWorker: [16, sy.VirtualWorker.simplify],
    str: [18, _simplify_str],
    pointers.ObjectWrapper: [19, sy.ObjectWrapper.simplify],
    GetNotPermittedError: [20, sy.exceptions.GetNotPermittedError.simplify],
    ResponseSignatureError: [20, sy.exceptions.ResponseSignatureError.simplify],
    torch.jit.ScriptModule: [21, _simplify_script_module],
    torch.jit.TopLevelTracedModule: [
        21,
        _simplify_script_module,
    ],  # treat as torch.jit.ScriptModule
    TrainConfig: [22, sy.TrainConfig.simplify],
}


forced_full_simplifiers = {VirtualWorker: [19, _force_full_simplify]}


def _detail(worker: AbstractWorker, obj: object) -> object:
    """
    This function reverses the functionality of _simplify. Where applicable,
    it converts simple objects into more complex objects such as converting
    binary objects into torch tensors. Read _simplify for more information on
    why _simplify and _detail are needed.

    Args:
        worker: the worker which is acquiring the message content, for example
        used to specify the owner of a tensor received(not obvious for
        virtual workers)
        obj: a simple Python object which msgpack deserialized

    Returns:
        obj: a more complex Python object which msgpack would have had trouble
            deserializing directly.

    """

    if type(obj) in (list, tuple):
        return detailers[obj[0]](worker, obj[1])
    else:
        return obj


detailers = [
    _detail_torch_tensor,
    _detail_torch_parameter,
    _detail_collection_tuple,
    _detail_collection_list,
    _detail_collection_set,
    _detail_dictionary,
    _detail_range,
    _detail_ndarray,
    _detail_slice,
    _detail_ellipsis,
    _detail_torch_device,
    sy.PointerTensor.detail,
    sy.LoggingTensor.detail,
    sy.AdditiveSharingTensor.detail,
    sy.MultiPointerTensor.detail,
    sy.Plan.detail,
    sy.VirtualWorker.detail,
    force_full_detail,
    _detail_str,
    sy.ObjectWrapper.detail,
    sy.exceptions.GetNotPermittedError.detail,
    _detail_script_module,
    sy.TrainConfig.detail,
]
