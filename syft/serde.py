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
from tempfile import TemporaryFile
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
import warnings
import zstd

import syft
import syft as sy

from syft.federated import TrainConfig

from syft.workers import AbstractWorker
from syft.workers import VirtualWorker

from syft.federated import Plan

from syft.exceptions import CompressionNotFoundException
from syft.exceptions import GetNotPermittedError

from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters import MultiPointerTensor
from syft.frameworks.torch.tensors.interpreters import PointerTensor
from syft.frameworks.torch.tensors.interpreters.abstract import initialize_tensor


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


def deserialize(binary: bin, worker: AbstractWorker = None, detail=True) -> object:
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
        worker = syft.torch.hook.local_worker

    # 1) Decompress the binary if needed
    binary = _decompress(binary)

    # 2) Deserialize
    # This function converts the binary into the appropriate python
    # object (or nested dict/collection of python objects)
    simple_objects = msgpack.loads(binary)

    if detail:
        # 3) Detail
        # This function converts typed, simple objects into their more
        # complex (and difficult to serialize) counterparts which the
        # serialization library wasn't natively able to serialize (such
        # as msgpack's inability to serialize torch tensors or ... or
        # python slice objects
        return _detail(worker, simple_objects)

    else:
        # sometimes we want to skip detailing (such as in Plan)
        return simple_objects


def _serialize_tensor(tensor) -> bin:
    """Serialize the tensor using as default Torch serialization strategy
    This function can be overridden to provide different tensor serialization strategies

    Args
        (torch.Tensor): an input tensor to be serialized

    Returns
        A serialized version of the input tensor

    """
    return torch_tensor_serializer(tensor)


def _deserialize_tensor(tensor_bin) -> torch.Tensor:
    """Deserialize the input tensor passed as parameter into a Torch tensor.
    This function can be overridden to provide different deserialization strategies

    Args
        tensor_bin: A binary representation of a tensor

    Returns
        a Torch tensor
    """
    return torch_tensor_deserializer(tensor_bin)


def numpy_tensor_serializer(tensor: torch.Tensor) -> bin:
    """Strategy to serialize a tensor using numpy npy format.
    If tensor requires to calculate gradients, it will detached.
    """
    if tensor.requires_grad:
        warnings.warn(
            "Torch to Numpy serializer can only be used with tensors that do not require grad. "
            "Detaching tensor to continue"
        )
        tensor = tensor.detach()

    np_tensor = tensor.numpy()
    outfile = TemporaryFile()
    numpy.save(outfile, np_tensor)
    # Simulate close and open by calling seek
    outfile.seek(0)
    return outfile.read()


def numpy_tensor_deserializer(tensor_bin) -> torch.Tensor:
    """"Strategy to deserialize a binary input in npy format into a Torch tensor"""
    input_file = TemporaryFile()
    input_file.write(tensor_bin)
    # read data from file
    input_file.seek(0)
    return torch.from_numpy(numpy.load(input_file))


def torch_tensor_serializer(tensor) -> bin:
    """Strategy to serialize a tensor using Torch saver"""
    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    return binary_stream.getvalue()


def torch_tensor_deserializer(tensor_bin) -> torch.Tensor:
    """"Strategy to deserialize a binary input using Torch load"""
    bin_tensor_stream = io.BytesIO(tensor_bin)
    return torch.load(bin_tensor_stream)


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


# Simplify/Detail Torch Tensors


def _simplify_torch_tensor(tensor: torch.Tensor) -> bin:
    """
    This function converts a torch tensor into a serliaized torch tensor
    using pickle. We choose to use this because PyTorch has a custom and
    very fast PyTorch pickler.

    Args:
        tensor (torch.Tensor): an input tensor to be serialized

    Returns:
        tuple: serialized tuple of torch tensor. The first value is the
        id of the tensor and the second is the binary for the PyTorch
        object. The third is the chain of abstractions, and the fourth
        (optinally) is the chain of graident tensors (nested tuple)
    """

    tensor_bin = _serialize_tensor(tensor)

    # note we need to do this expicitly because torch.save does not
    # seem to be including .grad by default

    if tensor.grad is not None:
        if hasattr(tensor, "child"):
            if isinstance(tensor.child, PointerTensor):
                grad_chain = None
            else:
                grad_chain = _simplify_torch_tensor(tensor.grad)
        else:
            grad_chain = _simplify_torch_tensor(tensor.grad)

    else:
        grad_chain = None

    chain = None

    # I think the pointer bug is is between here

    if hasattr(tensor, "child"):
        chain = _simplify(tensor.child)

    # and here... leaving a reerence here so i can find it later
    # TODO fix pointer bug

    tags = tensor.tags
    if tags is not None:
        tags = list(tags)
    return (tensor.id, tensor_bin, chain, grad_chain, tags, tensor.description)


def _detail_torch_tensor(worker: AbstractWorker, tensor_tuple: tuple) -> torch.Tensor:
    """
    This function converts a serialized torch tensor into a torch tensor
    using pickle.

    Args:
        tensor_tuple (bin): serialized obj of torch tensor. It's a tuple where
            the first value is the ID, the second vlaue is the binary for the
            PyTorch object, the third value is the chain of tensor abstractions,
            and the fourth object is the chain of gradients (.grad.grad, etc.)

    Returns:
        torch.Tensor: a torch tensor that was serialized
    """

    tensor_id, tensor_bin, chain, grad_chain, tags, description = tensor_tuple

    tensor = _deserialize_tensor(tensor_bin)

    # note we need to do this explicitly because torch.load does not
    # include .grad informatino
    if grad_chain is not None:
        tensor.grad = _detail_torch_tensor(worker, grad_chain)

    initialize_tensor(
        hook_self=syft.torch.hook,
        cls=tensor,
        torch_tensor=True,
        owner=worker,
        id=tensor_id,
        init_args=[],
        kwargs={},
    )

    if tags is not None:
        for i in range(len(tags)):
            tag = tags[i]
            if isinstance(tag, bytes):
                tag = tag.decode("utf-8")
            tags[i] = tag
        tensor.tags = tags

    if description is not None:
        if isinstance(description, bytes):
            description = description.decode("utf-8")
        tensor.description = description

    if chain is not None:
        chain = _detail(worker, chain)
        tensor.child = chain
        tensor.is_wrapper = True

    return tensor


# Simplify/Detail Parameters


def _simplify_torch_parameter(param: torch.nn.Parameter) -> bin:
    """
    This function converts a torch Parameter into a serialized torch Parameter

    Args:
        param (torch.nn.Parameter): an input Parameter to be serialized

    Returns:
        tuple: serialized tuple of torch Parameter. The first value is the
        id of the Parameter and the second is the binary for the PyTorch
        tensor data attribute and last is the requires_grad attr.
    """

    tensor = param.data

    tensor_ser = _simplify_torch_tensor(tensor)

    grad = param.grad

    if grad is not None and not (
        hasattr(grad, "child") and isinstance(grad.child, sy.PointerTensor)
    ):
        grad_ser = _simplify_torch_tensor(grad)
    else:
        grad_ser = None

    return (param.id, tensor_ser, param.requires_grad, grad_ser)


def _detail_torch_parameter(worker: AbstractWorker, param_tuple: tuple) -> torch.nn.Parameter:
    """
    This function converts a serialized torch Parameter into a torch Parameter.

    Args:
        param_tuple (tuple): serialized obj of torch tensor. It's a tuple where
            the first value is the ID and the second value is the binary for the
            PyTorch data attribute et and third value is the requires_grad attr.

    Returns:
        torch.Parameter: a torch Parameter that was serialized
    """
    param_id, tensor_ser, requires_grad, grad_ser = param_tuple

    tensor = _detail_torch_tensor(worker, tensor_ser)

    if grad_ser is not None:
        grad = _detail_torch_tensor(worker, grad_ser)
        grad.garbage_collect_data = False
    elif hasattr(tensor, "child") and isinstance(tensor.child, sy.PointerTensor):
        grad = tensor.attr("grad")
    else:
        grad = None

    param = torch.nn.Parameter(tensor, requires_grad)
    param.id = param_id
    param.grad = grad

    return param


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
        my_collection (Collection): a collection of python objects

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


def _detail_collection_list(worker: AbstractWorker, my_collection: Collection) -> Collection:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a collection of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_collection (Collection): a collection of simple python objects (including binary).

    Returns:
        Collection: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        try:
            pieces.append(_detail(worker, part).decode("utf-8"))  # transform bytes back to string
        except AttributeError:
            pieces.append(_detail(worker, part))

    return pieces


def _detail_collection_set(worker: AbstractWorker, my_collection: Collection) -> Collection:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a collection of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_collection (Collection): a collection of simple python objects (including binary).

    Returns:
        Collection: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_collection:
        try:
            pieces.append(_detail(worker, part).decode("utf-8"))  # transform bytes back to string
        except AttributeError:
            pieces.append(_detail(worker, part))
    return set(pieces)


def _detail_collection_tuple(worker: AbstractWorker, my_tuple: Tuple) -> Tuple:
    """
    This function is designed to operate in the opposite direction of
    _simplify_collection. It takes a tuple of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.
    This is only applicable to tuples. They need special handling because
    `msgpack` is encoding a tuple as a list.

    Args:
        worker: the worker doing the deserialization
        my_tuple (Tuple): a collection of simple python objects (including binary).

    Returns:
        tuple: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """

    pieces = list()

    # Step 1: deserialize each part of the collection
    for part in my_tuple:
        pieces.append(_detail(worker, part))

    return tuple(pieces)


# Dictionaries


def _simplify_dictionary(my_dict: Dict) -> Dict:
    """
    This function is designed to search a dict for any objects
    which may need to be simplified (i.e., torch tensors). It iterates
    through each key, value in the dict and calls _simplify on it. Finally,
    it returns the output dict as the same type as the input dict
    so that the consuming serialization step knows the correct type info. The
    reverse function to this function is _detail_dictionary, which undoes
    the functionality of this function.

    Args:
        my_dict (Dict): a dictionary of python objects

    Returns:
        Dict: a dictionary of the same type as the input of simplified
            objects.

    """
    pieces = list()
    # for dictionaries we want to simplify both the key and the value
    for key, value in my_dict.items():
        pieces.append((_simplify(key), _simplify(value)))

    return pieces


def _detail_dictionary(worker: AbstractWorker, my_dict: Dict) -> Dict:
    """
    This function is designed to operate in the opposite direction of
    _simplify_dictionary. It takes a dictionary of simple python objects
    and iterates through it to determine whether objects in the collection
    need to be converted into more advanced types. In particular, it
    converts binary objects into torch Tensors where appropriate.

    Args:
        worker: the worker doing the deserialization
        my_dict (Dict): a dictionary of simple python objects (including binary).

    Returns:
        tuple: a collection of the same type as the input where the objects
            in the collection have been detailed.
    """
    pieces = {}
    # for dictionaries we want to detail both the key and the value
    for key, value in my_dict:
        detailed_key = _detail(worker, key)
        try:
            detailed_key = detailed_key.decode("utf-8")
        except AttributeError:
            pass

        detailed_value = _detail(worker, value)
        try:
            detailed_value = detailed_value.decode("utf-8")
        except AttributeError:
            pass

        pieces[detailed_key] = detailed_value

    return pieces


# Range


def _simplify_range(my_range: range) -> Tuple[int, int, int]:
    """
    This function extracts the start, stop and step from the range.

    Args:
        my_range (range): a range object

    Returns:
        list: a list defining the range parameters [start, stop, step]

    Examples:

        range_parameters = _simplify_range(range(1, 3, 4))

        assert range_parameters == [1, 3, 4]

    """

    return (my_range.start, my_range.stop, my_range.step)


def _detail_range(worker: AbstractWorker, my_range_params: Tuple[int, int, int]) -> range:
    """
    This function extracts the start, stop and step from a tuple.

    Args:
        worker: the worker doing the deserialization (only here to standardise signature
            with other _detail functions)
        my_range_params (tuple): a tuple defining the range parameters [start, stop, step]

    Returns:
        range: a range object

    Examples:
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
        my_array (numpy.ndarray): a numpy array

    Returns:
        list: a list holding the byte representation, shape and dtype of the array

    Examples:

        arr_representation = _simplify_ndarray(numpy.random.random([1000, 1000])))

    """
    arr_bytes = my_array.tobytes()
    arr_shape = my_array.shape
    arr_dtype = my_array.dtype.name

    return (arr_bytes, arr_shape, arr_dtype)


def _detail_ndarray(
    worker: AbstractWorker, arr_representation: Tuple[bin, Tuple, str]
) -> numpy.ndarray:
    """
    This function reconstruct a numpy array from it's byte data, the shape and the dtype
        by first loading the byte data with the appropiate dtype and then reshaping it into the
        original shape

    Args:
        worker: the worker doing the deserialization
        arr_representation (tuple): a tuple holding the byte representation, shape
        and dtype of the array

    Returns:
        numpy.ndarray: a numpy array

    Examples:
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
        my_slice (slice): a python slice

    Returns:
        tuple : a list holding the start, stop and step values

    Examples:

        slice_representation = _simplify_slice(slice(1,2,3))

    """
    return (my_slice.start, my_slice.stop, my_slice.step)


def _detail_slice(worker: AbstractWorker, my_slice: Tuple[int, int, int]) -> slice:
    """
    This function extracts the start, stop and step from a list.

    Args:
        my_slice (tuple): a list defining the slice parameters [start, stop, step]

    Returns:
        range: a range object

    Examples:
        new_range = _detail_range([1, 3, 4])

        assert new_range == range(1, 3, 4)

    """

    return slice(my_slice[0], my_slice[1], my_slice[2])


def _simplify_ellipsis(e: Ellipsis) -> bytes:
    return b""


def _simplify_torch_device(device: torch.device) -> Tuple[str]:
    return device.type


def _detail_ellipsis(worker: AbstractWorker, ellipsis: bytes) -> Ellipsis:
    return ...


def _detail_torch_device(worker: AbstractWorker, device_type: str) -> torch.device:
    return torch.device(type=device_type)


def _simplify_pointer_tensor(ptr: PointerTensor) -> tuple:
    """
    This function takes the attributes of a PointerTensor and saves them in a dictionary
    Args:
        ptr (PointerTensor): a PointerTensor
    Returns:
        tuple: a tuple holding the unique attributes of the pointer
    Examples:
        data = _simplify_pointer_tensor(ptr)
    """

    return (
        ptr.id,
        ptr.id_at_location,
        ptr.location.id,
        ptr.point_to_attr,
        ptr._shape,
        ptr.garbage_collect_data,
    )

    # a more general but slower/more verbose option

    # data = vars(ptr).copy()
    # for k, v in data.items():
    #     if isinstance(v, AbstractWorker):
    #         data[k] = v.id
    # return _simplify_dictionary(data)


def _detail_pointer_tensor(worker: AbstractWorker, tensor_tuple: tuple) -> PointerTensor:
    """
    This function reconstructs a PointerTensor given it's attributes in form of a dictionary.
    We use the spread operator to pass the dict data as arguments
    to the init method of PointerTensor
    Args:
        worker: the worker doing the deserialization
        tensor_tuple: a tuple holding the attributes of the PointerTensor
    Returns:
        PointerTensor: a PointerTensor
    Examples:
        ptr = _detail_pointer_tensor(data)
    """
    # TODO: fix comment for this and simplifier
    obj_id, id_at_location, worker_id, point_to_attr, shape, garbage_collect_data = tensor_tuple

    if isinstance(worker_id, bytes):
        worker_id = worker_id.decode()

    if shape is not None:
        shape = torch.Size(shape)

    # If the pointer received is pointing at the current worker, we load the tensor instead
    if worker_id == worker.id:
        tensor = worker.get_obj(id_at_location)

        if point_to_attr is not None and tensor is not None:

            point_to_attrs = point_to_attr.decode("utf-8").split(".")
            for attr in point_to_attrs:
                if len(attr) > 0:
                    tensor = getattr(tensor, attr)

            if tensor is not None:

                if not tensor.is_wrapper and not isinstance(tensor, torch.Tensor):
                    # if the tensor is a wrapper then it doesn't need to be wrapped
                    # i the tensor isn't a wrapper, BUT it's just a plain torch tensor,
                    # then it doesn't need to be wrapped.
                    # if the tensor is not a wrapper BUT it's also not a torch tensor,
                    # then it needs to be wrapped or else it won't be able to be used
                    # by other interfaces
                    tensor = tensor.wrap()

        return tensor
    # Else we keep the same Pointer
    else:

        location = syft.torch.hook.local_worker.get_worker(worker_id)

        ptr = PointerTensor(
            location=location,
            id_at_location=id_at_location,
            owner=worker,
            id=obj_id,
            shape=shape,
            garbage_collect_data=garbage_collect_data,
        )

        return ptr

    # a more general but slower/more verbose option

    # new_data = {}
    # for k, v in data.items():
    #     key = k.decode()
    #     if type(v) is bytes:
    #         val_str = v.decode()
    #         val = syft.local_worker.get_worker(val_str)
    #     else:
    #         val = v
    #     new_data[key] = val
    # return PointerTensor(**new_data)


def _simplify_log_tensor(tensor: LoggingTensor) -> tuple:
    """
    This function takes the attributes of a LogTensor and saves them in a tuple
    Args:
        tensor (LoggingTensor): a LogTensor
    Returns:
        tuple: a tuple holding the unique attributes of the log tensor
    Examples:
        data = _simplify_log_tensor(tensor)
    """

    chain = None
    if hasattr(tensor, "child"):
        chain = _simplify(tensor.child)
    return (tensor.id, chain)


def _detail_log_tensor(worker: AbstractWorker, tensor_tuple: tuple) -> LoggingTensor:
    """
    This function reconstructs a LogTensor given it's attributes in form of a tuple.
    Args:
        worker: the worker doing the deserialization
        tensor_tuple: a tuple holding the attributes of the LogTensor
    Returns:
        LoggingTensor: a LogTensor
    Examples:
        logtensor = _detail_log_tensor(data)
    """
    obj_id, chain = tensor_tuple

    tensor = LoggingTensor(owner=worker, id=obj_id)

    if chain is not None:
        chain = _detail(worker, chain)
        tensor.child = chain

    return tensor


def _simplify_additive_shared_tensor(tensor: AdditiveSharingTensor) -> tuple:
    """
    This function takes the attributes of a AdditiveSharingTensor and saves them in a tuple
    Args:
        tensor (AdditiveSharingTensor): a AdditiveSharingTensor
    Returns:
        tuple: a tuple holding the unique attributes of the additive shared tensor
    Examples:
        data = _simplify_additive_shared_tensor(tensor)
    """

    chain = None
    if hasattr(tensor, "child"):
        chain = _simplify(tensor.child)
    return (tensor.id, tensor.field, tensor.crypto_provider.id, chain)


def _detail_additive_shared_tensor(
    worker: AbstractWorker, tensor_tuple: tuple
) -> AdditiveSharingTensor:
    """
        This function reconstructs a AdditiveSharingTensor given it's attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the AdditiveSharingTensor
        Returns:
            AdditiveSharingTensor: a AdditiveSharingTensor
        Examples:
            shared_tensor = _detail_additive_shared_tensor(data)
        """

    tensor_id, field, crypto_provider, chain = tensor_tuple

    tensor = AdditiveSharingTensor(
        owner=worker, id=tensor_id, field=field, crypto_provider=worker.get_worker(crypto_provider)
    )

    if chain is not None:
        chain = _detail(worker, chain)
        tensor.child = chain

    return tensor


def _simplify_multi_pointer_tensor(tensor: MultiPointerTensor) -> tuple:
    """
    This function takes the attributes of a MultiPointerTensor and saves them in a tuple
    Args:
        tensor (MultiPointerTensor): a MultiPointerTensor
    Returns:
        tuple: a tuple holding the unique attributes of the additive shared tensor
    Examples:
        data = _simplify_additive_shared_tensor(tensor)
    """

    chain = None
    if hasattr(tensor, "child"):
        chain = _simplify(tensor.child)
    return (tensor.id, chain)


def _detail_multi_pointer_tensor(worker: AbstractWorker, tensor_tuple: tuple) -> MultiPointerTensor:
    """
        This function reconstructs a MultiPointerTensor given it's attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the MultiPointerTensor
        Returns:
            MultiPointerTensor: a MultiPointerTensor
        Examples:
            multi_pointer_tensor = _detail_multi_pointer_tensor(data)
        """

    tensor_id, chain = tensor_tuple

    tensor = MultiPointerTensor(owner=worker, id=tensor_id)

    if chain is not None:
        chain = _detail(worker, chain)
        tensor.child = chain

    return tensor


def _simplify_train_config(train_config: TrainConfig) -> tuple:
    """
    This function takes the attributes of a TrainConfig and saves them in a tuple
    Args:
        train_config: a TrainConfig object
    Returns:
        tuple: a tuple holding the unique attributes of the TrainConfig object
    """
    return (
        _simplify(train_config.loss_plan),
        _simplify(train_config.forward_plan),
        train_config.batch_size,
        train_config.epochs,
        _simplify(train_config.optimizer),
        train_config.lr,
        _simplify(train_config.id),
    )


def _detail_train_config(worker: AbstractWorker, train_config_tuple: tuple) -> tuple:
    """This function reconstructs a TrainConfig object given it's attributes in the form of a tuple.
    Args:
        worker: the worker doing the deserialization
        train_config_tuple: a tuple holding the attributes of the TrainConfig
    Returns:
        Plan: a Plan object
    """

    loss_plan, forward_plan, batch_size, epochs, optimizer, lr, id = train_config_tuple

    id = _detail(worker, id)
    detailed_loss_plan = _detail(worker, loss_plan)
    detailed_forward_plan = _detail(worker, forward_plan)
    detailed_optimizer = _detail(worker, optimizer)

    train_config = syft.TrainConfig(
        owner=worker,
        id=id,
        forward_plan=detailed_forward_plan,
        loss_plan=detailed_loss_plan,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=detailed_optimizer,
        lr=lr,
    )

    return train_config


def _simplify_plan(plan: Plan) -> tuple:
    """
    This function takes the attributes of a Plan and saves them in a tuple
    Args:
        plan (Plan): a Plan object
    Returns:
        tuple: a tuple holding the unique attributes of the Plan object

    """
    readable_plan = _simplify(plan.readable_plan)
    return (
        readable_plan,
        _simplify(plan.id),
        _simplify(plan.arg_ids),
        _simplify(plan.result_ids),
        _simplify(plan.name),
        _simplify(plan.tags),
        _simplify(plan.description),
    )


def _detail_plan(worker: AbstractWorker, plan_tuple: tuple) -> Plan:
    """This function reconstructs a Plan object given it's attributes in the form of a tuple.
    Args:
        worker: the worker doing the deserialization
        plan_tuple: a tuple holding the attributes of the Plan
    Returns:
        Plan: a Plan object
    """

    readable_plan, id, arg_ids, result_ids, name, tags, description = plan_tuple
    id = _detail(worker, id)
    arg_ids = _detail(worker, arg_ids)
    result_ids = _detail(worker, result_ids)

    plan = syft.Plan(
        owner=worker,
        id=id,
        arg_ids=arg_ids,
        result_ids=result_ids,
        readable_plan=_detail(worker, readable_plan),
    )

    plan.name = _detail(worker, name)
    plan.tags = _detail(worker, tags)
    plan.description = _detail(worker, description)

    return plan


def _simplify_worker(worker: AbstractWorker) -> tuple:
    """

    """

    return (_simplify(worker.id),)


def _detail_worker(worker: AbstractWorker, worker_tuple: tuple) -> PointerTensor:
    """
    This function reconstructs a PlanPointer given it's attributes in form of a tuple.

    Args:
        worker: the worker doing the deserialization
        plan_pointer_tuple: a tuple holding the attributes of the PlanPointer
    Returns:
        PointerTensor: a PointerTensor
    Examples:
        ptr = _detail_pointer_tensor(data)
    """
    worker_id = _detail(worker, worker_tuple[0])

    referenced_worker = worker.get_worker(worker_id)

    return referenced_worker


def _simplify_GetNotPermittedError(error: GetNotPermittedError) -> tuple:
    """Simplifies a GetNotPermittedError into its message"""
    return (getattr(error, "message", str(error)),)


def _detail_GetNotPermittedError(
    worker: AbstractWorker, error_tuple: tuple
) -> GetNotPermittedError:
    """Details and raises a GetNotPermittedError

    Args:
        worker: the worker doing the deserialization
        error_tuple: a tuple holding the message of the GetNotPermittedError
    Raises:
        GetNotPermittedError: the error thrown when get is not permitted
    """

    raise GetNotPermittedError(error_tuple[0])


def _force_full_simplify_worker(worker: AbstractWorker) -> tuple:
    """

    """

    return (_simplify(worker.id), _simplify(worker._objects), worker.auto_add)


def _force_full_detail_worker(worker: AbstractWorker, worker_tuple: tuple) -> tuple:
    worker_id, _objects, auto_add = worker_tuple
    worker_id = _detail(worker, worker_id)

    result = sy.VirtualWorker(sy.hook, worker_id, auto_add=auto_add)
    _objects = _detail(worker, _objects)
    result._objects = _objects

    # make sure they weren't accidentally double registered
    for _, obj in _objects.items():
        if obj.id in worker._objects:
            del worker._objects[obj.id]

    return result


def _simplify_str(obj: str) -> tuple:
    return (obj.encode("utf-8"),)


def _detail_str(worker: AbstractWorker, str_tuple: tuple) -> str:
    return str_tuple[0].decode("utf-8")


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
    PointerTensor: [11, _simplify_pointer_tensor],
    LoggingTensor: [12, _simplify_log_tensor],
    AdditiveSharingTensor: [13, _simplify_additive_shared_tensor],
    MultiPointerTensor: [14, _simplify_multi_pointer_tensor],
    Plan: [15, _simplify_plan],
    VirtualWorker: [16, _simplify_worker],
    GetNotPermittedError: [17, _simplify_GetNotPermittedError],
    str: [18, _simplify_str],
    TrainConfig: [19, _simplify_train_config],
}

forced_full_simplifiers = {VirtualWorker: [20, _force_full_simplify_worker]}


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
    _detail_pointer_tensor,
    _detail_log_tensor,
    _detail_additive_shared_tensor,
    _detail_multi_pointer_tensor,
    _detail_plan,
    _detail_worker,
    _detail_GetNotPermittedError,
    _detail_str,
    _detail_train_config,
    _force_full_detail_worker,
]
