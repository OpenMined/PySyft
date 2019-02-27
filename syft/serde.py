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
import syft
import syft as sy

from syft.workers import AbstractWorker
from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.frameworks.torch.tensors.interpreters import PointerTensor

from syft.frameworks.torch.tensors.interpreters.abstract import initialize_tensor
from syft.exceptions import CompressionNotFoundException


# COMPRESSION SCHEME INT CODES
LZ4 = 0
ZSTD = 1


# Indicator on binary header that compression was not used.
UNUSED_COMPRESSION_INDICATOR = 48


# High Level Public Functions (these are the ones you use)
def serialize(obj: object, compress=True, compress_scheme=LZ4) -> bin:
    """This method can serialize any object PySyft needs to send or store.

    This is the high level function for serializing any object or collection
    of objects which PySyft needs to send over the wire. It includes three
    steps, Simplify, Serialize, and Compress as described inline below.

    Args:
        obj (object): the object to be serialized
        compress (bool): whether or not to compress the object
        compress_scheme (int): the integer code specifying which compression
            scheme to use (see above this method for scheme codes) if
            compress == True.

    Returns:
        binary: the serialized form of the object.

    """
    # 1) Simplify
    # simplify difficult-to-serialize objects. See the _simpliy method
    # for details on how this works. The general purpose is to handle types
    # which the fast serializer cannot handle
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
        compress_stream = _compress(binary, compress_scheme)
        if len(compress_stream) < len(binary):
            return b"\x31" + compress_stream

    return b"\x30" + binary


def deserialize(
    binary: bin, worker: AbstractWorker = None, compressed=True, compress_scheme=LZ4
) -> object:
    """ This method can deserialize any object PySyft needs to send or store.

    This is the high level function for deserializing any object or collection
    of objects which PySyft has sent over the wire or stored. It includes three
    steps, Decompress, Deserialize, and Detail as described inline below.

    Args:
        binary (bin): the serialized object to be deserialized.
        worker (AbstractWorker): the worker which is acquiring the message content, for example
            used to specify the owner of a tensor received(not obvious for
            virtual workers)
        compressed (bool): whether or not the serialized object is compressed
            (and thus whether or not it needs to be decompressed).
        compress_scheme (int): the integer code specifying which compression
            scheme was used if decompression is needed (see above this method
            for scheme codes).

    Returns:
        binary: the serialized form of the object.
    """
    if worker is None:
        worker = syft.torch.hook.local_worker

    # check the 1-byte header to see if input stream was compressed or not
    if binary[0] == UNUSED_COMPRESSION_INDICATOR:
        compressed = False

    # remove the 1-byte header from the input stream
    binary = binary[1:]
    # 1)  Decompress
    # If enabled, this functionality decompresses the binary
    if compressed:
        binary = _decompress(binary, compress_scheme)

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
    return _detail(worker, simple_objects)


# Chosen Compression Algorithm


def _compress(decompressed_input_bin: bin, compress_scheme=LZ4) -> bin:
    """
    This function compresses a binary using LZ4

    Args:
        decompressed_input_bin (bin): binary to be compressed
        compress_scheme: the compression method to use

    Returns:
        bin: a compressed binary

    """
    if compress_scheme == LZ4:
        return lz4.frame.compress(decompressed_input_bin)
    elif compress_scheme == ZSTD:
        return zstd.compress(decompressed_input_bin)
    else:
        raise CompressionNotFoundException(
            "compression scheme note found for" " compression code:" + str(compress_scheme)
        )


def _decompress(compressed_input_bin: bin, compress_scheme=LZ4) -> bin:
    """
    This function decompresses a binary using LZ4

    Args:
        compressed_input_bin (bin): a compressed binary
        compress_scheme: the compression method to use

    Returns:
        bin: decompressed binary

    """
    if compress_scheme == LZ4:
        return lz4.frame.decompress(compressed_input_bin)
    elif compress_scheme == ZSTD:
        return zstd.decompress(compressed_input_bin)
    else:
        raise CompressionNotFoundException(
            "compression scheme note found for" " compression code:" + str(compress_scheme)
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

    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    tensor_bin = binary_stream.getvalue()

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

    bin_tensor_stream = io.BytesIO(tensor_bin)
    tensor = torch.load(bin_tensor_stream)

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
            tags[i] = tags[i].decode("utf-8")
        tensor.tags = tags

    if description is not None:
        tensor.description = description.decode("utf-8")

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
    pieces = {}
    # for dictionaries we want to simplify both the key and the value
    for key, value in my_dict.items():
        pieces[_simplify(key)] = _simplify(value)

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
    for key, value in my_dict.items():

        try:
            detailed_key = _detail(worker, key).decode("utf-8")
        except AttributeError:
            detailed_key = _detail(worker, key)

        try:
            detailed_value = _detail(worker, value).decode("utf-8")
        except AttributeError:
            detailed_value = _detail(worker, value)

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

    return (ptr.id, ptr.id_at_location, ptr.location.id, ptr.point_to_attr, ptr.shape)

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
    obj_id = tensor_tuple[0]
    id_at_location = tensor_tuple[1]
    worker_id = tensor_tuple[2].decode("utf-8")
    point_to_attr = tensor_tuple[3]
    shape = tensor_tuple[4]

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
            garbage_collect_data=True,
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
}


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

    if type(obj) == list:
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
]
