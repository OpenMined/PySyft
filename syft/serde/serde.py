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
If different compressions are required, the worker can override the function apply_compress_scheme
"""
from collections import OrderedDict

import inspect
import lz4
from lz4 import (  # noqa: F401
    frame,
)  # needed as otherwise we will get: module 'lz4' has no attribute 'frame'
import msgpack
import zstd

import syft
from syft import dependency_check
from syft.federated.train_config import TrainConfig
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.crt_precision import CRTPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.messaging.plan import Plan
from syft.messaging.message import Message
from syft.messaging.message import Operation
from syft.messaging.message import ObjectMessage
from syft.messaging.message import ObjectRequestMessage
from syft.messaging.message import IsNoneMessage
from syft.messaging.message import GetShapeMessage
from syft.messaging.message import ForceObjectDeleteMessage
from syft.messaging.message import SearchMessage
from syft.messaging.message import PlanCommandMessage
from syft.serde.native_serde import MAP_NATIVE_SIMPLIFIERS_AND_DETAILERS
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker

from syft.exceptions import CompressionNotFoundException
from syft.exceptions import GetNotPermittedError
from syft.exceptions import ResponseSignatureError

if dependency_check.torch_available:
    from syft.serde.torch_serde import MAP_TORCH_SIMPLIFIERS_AND_DETAILERS
else:
    MAP_TORCH_SIMPLIFIERS_AND_DETAILERS = {}

if dependency_check.tensorflow_available:
    from syft_tensorflow.serde import MAP_TF_SIMPLIFIERS_AND_DETAILERS
else:
    MAP_TF_SIMPLIFIERS_AND_DETAILERS = {}

# Maps a type to a tuple containing its simplifier and detailer function
MAP_TO_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    list(MAP_NATIVE_SIMPLIFIERS_AND_DETAILERS.items())
    + list(MAP_TORCH_SIMPLIFIERS_AND_DETAILERS.items())
    + list(MAP_TF_SIMPLIFIERS_AND_DETAILERS.items())
)

# If an object implements its own simplify and detail functions it should be stored in this list
OBJ_SIMPLIFIER_AND_DETAILERS = [
    AdditiveSharingTensor,
    FixedPrecisionTensor,
    CRTPrecisionTensor,
    LoggingTensor,
    MultiPointerTensor,
    Plan,
    PointerTensor,
    ObjectWrapper,
    TrainConfig,
    BaseWorker,
    AutogradTensor,
    Message,
    Operation,
    ObjectMessage,
    ObjectRequestMessage,
    IsNoneMessage,
    GetShapeMessage,
    ForceObjectDeleteMessage,
    SearchMessage,
    PlanCommandMessage,
]

# If an object implements its own force_simplify and force_detail functions it should be stored in this list
OBJ_FORCE_FULL_SIMPLIFIER_AND_DETAILERS = [BaseWorker]

# For registering syft objects with custom simplify and detail methods
EXCEPTION_SIMPLIFIER_AND_DETAILERS = [GetNotPermittedError, ResponseSignatureError]

# COMPRESSION SCHEME INT CODES
NO_COMPRESSION = 40
LZ4 = 41
ZSTD = 42
scheme_to_bytes = {
    NO_COMPRESSION: NO_COMPRESSION.to_bytes(1, byteorder="big"),
    LZ4: LZ4.to_bytes(1, byteorder="big"),
    ZSTD: ZSTD.to_bytes(1, byteorder="big"),
}

## SECTION: High Level Simplification Router
def _force_full_simplify(obj: object) -> object:
    """To force a full simplify generally if the usual _simplify is not suitable.

    If we can not full simplify a object we simplify it as usual instead.

    Args:
        obj: The object.

    Returns:
        The simplified object.
    """
    # check to see if there is a full simplifier
    # for this type. If there is, return the full simplified object.
    current_type = type(obj)
    if current_type in forced_full_simplifiers:
        result = (
            forced_full_simplifiers[current_type][0],
            forced_full_simplifiers[current_type][1](obj),
        )
    # If we already tried to find a full simplifier for this type but failed, we should
    # simplify it instead.
    elif current_type in no_full_simplifiers_found:
        return _simplify(obj)
    else:
        # If the object type is not in forced_full_simplifiers,
        # we check the classes that this object inherits from.
        # `inspect.getmro` give us all types this object inherits
        # from, including `type(obj)`. We can skip the type of the
        # object because we already tried this in the
        # previous step.
        classes_inheritance = inspect.getmro(type(obj))[1:]

        for inheritance_type in classes_inheritance:
            if inheritance_type in forced_full_simplifiers:
                # Store the inheritance_type in forced_full_simplifiers so next
                # time we see this type serde will be faster.
                forced_full_simplifiers[current_type] = forced_full_simplifiers[inheritance_type]
                result = (
                    forced_full_simplifiers[current_type][0],
                    forced_full_simplifiers[current_type][1](obj),
                )
                return result

        # If there is not a full_simplifier for this
        # object, then we simplify it.
        no_full_simplifiers_found.add(current_type)
        return _simplify(obj)


## SECTION: dinamically generate simplifiers and detailers
def _generate_simplifiers_and_detailers():
    """Generate simplifiers, forced full simplifiers and detailers,
    by registering native and torch types, syft objects with custom
    simplify and detail methods, or syft objects with custom
    force_simplify and force_detail methods.

    Returns:
        The simplifiers, forced_full_simplifiers, detailers
    """
    simplifiers = OrderedDict()
    forced_full_simplifiers = OrderedDict()
    detailers = []

    def _add_simplifier_and_detailer(curr_type, simplifier, detailer, forced=False):
        if detailer in detailers:
            curr_index = detailers.index(detailer)
        else:
            curr_index = len(detailers)
            detailers.append(detailer)

        if forced:
            forced_full_simplifiers[curr_type] = (curr_index, simplifier)
        else:
            simplifiers[curr_type] = (curr_index, simplifier)

    # Register native and torch types
    for curr_type in MAP_TO_SIMPLIFIERS_AND_DETAILERS:
        simplifier, detailer = MAP_TO_SIMPLIFIERS_AND_DETAILERS[curr_type]
        _add_simplifier_and_detailer(curr_type, simplifier, detailer)

    # Register syft objects with custom simplify and detail methods
    for syft_type in OBJ_SIMPLIFIER_AND_DETAILERS + EXCEPTION_SIMPLIFIER_AND_DETAILERS:
        simplifier, detailer = syft_type.simplify, syft_type.detail
        _add_simplifier_and_detailer(syft_type, simplifier, detailer)

    # Register syft objects with custom force_simplify and force_detail methods
    for syft_type in OBJ_FORCE_FULL_SIMPLIFIER_AND_DETAILERS:
        force_simplifier, force_detailer = syft_type.force_simplify, syft_type.force_detail
        _add_simplifier_and_detailer(syft_type, force_simplifier, force_detailer, forced=True)

    return simplifiers, forced_full_simplifiers, detailers


simplifiers, forced_full_simplifiers, detailers = _generate_simplifiers_and_detailers()
# Store types that are not simplifiable (int, float, None) so we
# can ignore them during serialization.
no_simplifiers_found, no_full_simplifiers_found = set(), set()


## SECTION:  High Level Public Functions (these are the ones you use)
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
            simple_objects = _force_full_simplify(obj)
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
        details (bool): there are some cases where we need to perform the decompression
            and deserialization part, but we don't need to detail all the message.
            This is the case for Plan workers for instance

    Returns:
        object: the deserialized form of the binary input.
    """
    if worker is None:
        # TODO[jvmancuso]: This might be worth a standalone function.
        worker = syft.framework.hook.local_worker

    # 1) Decompress the binary if needed
    binary = _decompress(binary)

    # 2) Deserialize
    # This function converts the binary into the appropriate python
    # object (or nested dict/collection of python objects)
    simple_objects = msgpack.loads(binary, use_list=False)

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


## SECTION: chosen Compression Algorithm


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
        decompressed_input_bin: the binary to be compressed

    Returns:
        a tuple (compressed_result, LZ4)
    """
    return lz4.frame.compress(decompressed_input_bin), LZ4


def apply_zstd_compression(decompressed_input_bin) -> tuple:
    """
    Apply ZSTD compression to the input

    Args:
        decompressed_input_bin: the binary to be compressed

    Returns:
        a tuple (compressed_result, ZSTD)
    """

    return zstd.compress(decompressed_input_bin), ZSTD


def apply_no_compression(decompressed_input_bin) -> tuple:
    """
    No compression is applied to the input

    Args:
        decompressed_input_bin: the binary

    Returns:
        a tuple (the binary, LZ4)
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
    try:
        z = scheme_to_bytes[compress_scheme] + compress_stream
        return z
    except KeyError:
        raise CompressionNotFoundException(
            f"Compression scheme not found for compression code: {str(compress_scheme)}"
        )


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
            f"Compression scheme not found for compression code: {str(compress_scheme)}"
        )


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
        obj: An object which may need to be simplified.

    Returns:
        An simple Python object which msgpack can serialize.

    Raises:
        ValueError: if `move_this` or `in_front_of_that` are not both single ASCII
        characters.
    """

    # Check to see if there is a simplifier
    # for this type. If there is, return the simplified object.
    current_type = type(obj)
    if current_type in simplifiers:
        result = (simplifiers[current_type][0], simplifiers[current_type][1](obj))
        return result

    # If we already tried to find a simplifier for this type but failed, we should
    # just return the object as it is.
    elif current_type in no_simplifiers_found:
        return obj

    else:
        # If the object type is not in simplifiers,
        # we check the classes that this object inherits from.
        # `inspect.getmro` give us all types this object inherits
        # from, including `type(obj)`. We can skip the type of the
        # object because we already tried this in the
        # previous step.
        classes_inheritance = inspect.getmro(type(obj))[1:]

        for inheritance_type in classes_inheritance:
            if inheritance_type in simplifiers:
                # Store the inheritance_type in simplifiers so next time we see this type
                # serde will be faster.
                simplifiers[current_type] = simplifiers[inheritance_type]
                result = (simplifiers[current_type][0], simplifiers[current_type][1](obj))
                return result

        # if there is not a simplifier for this
        # object, then the object is already a
        # simple python object and we can just
        # return it.
        no_simplifiers_found.add(current_type)
        return obj


def _detail(worker: AbstractWorker, obj: object) -> object:
    """Reverses the functionality of _simplify.
    Where applicable, it converts simple objects into more complex objects such
    as converting binary objects into torch tensors. Read _simplify for more
    information on why _simplify and detail are needed.

    Args:
        worker: the worker which is acquiring the message content, for example
        used to specify the owner of a tensor received(not obvious for
        virtual workers).
        obj: a simple Python object which msgpack deserialized.

    Returns:
        obj: a more complex Python object which msgpack would have had trouble
            deserializing directly.
    """
    if type(obj) in (list, tuple):
        return detailers[obj[0]](worker, obj[1])
    else:
        return obj
