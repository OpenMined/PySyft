"""
This file exists to provide one common place for all serialisation and simplify_ and _detail
for all tensors (Torch) and collections.
"""
from tempfile import TemporaryFile
from typing import Collection
from typing import Dict
from typing import Tuple
import torch

import io
import numpy
import warnings

import syft
import syft as sy

from syft.federated import TrainConfig

from syft.workers import AbstractWorker
from syft.workers import VirtualWorker

from syft.federated import Plan

from syft.exceptions import GetNotPermittedError
from syft.exceptions import ResponseSignatureError

from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters import MultiPointerTensor
from syft.frameworks.torch.tensors.interpreters.abstract import initialize_tensor
from syft.frameworks.torch import pointers


from syft.serde.native_serde import (
    _simplify_slice,
    _simplify_ellipsis,
    _simplify_range,
    _simplify_str,
    _detail_slice,
    _detail_ellipsis,
    _detail_range,
    _detail_str,
)


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
            if isinstance(tensor.child, pointers.PointerTensor):
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
        chain = detail(worker, chain)
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
        hasattr(grad, "child") and isinstance(grad.child, pointers.PointerTensor)
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
    elif hasattr(tensor, "child") and isinstance(tensor.child, pointers.PointerTensor):
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
            pieces.append(detail(worker, part).decode("utf-8"))  # transform bytes back to string
        except AttributeError:
            pieces.append(detail(worker, part))

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
            pieces.append(detail(worker, part).decode("utf-8"))  # transform bytes back to string
        except AttributeError:
            pieces.append(detail(worker, part))
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
        pieces.append(detail(worker, part))

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
        detailed_key = detail(worker, key)
        try:
            detailed_key = detailed_key.decode("utf-8")
        except AttributeError:
            pass

        detailed_value = detail(worker, value)
        try:
            detailed_value = detailed_value.decode("utf-8")
        except AttributeError:
            pass

        pieces[detailed_key] = detailed_value

    return pieces


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


def _simplify_torch_device(device: torch.device) -> Tuple[str]:
    return device.type


def _detail_torch_device(worker: AbstractWorker, device_type: str) -> torch.device:
    return torch.device(type=device_type)


def _force_fullsimplify(worker: AbstractWorker) -> tuple:
    """

    """

    return (_simplify(worker.id), _simplify(worker._objects), worker.auto_add)


def force_full_detail(worker: AbstractWorker, worker_tuple: tuple) -> tuple:
    worker_id, _objects, auto_add = worker_tuple
    worker_id = detail(worker, worker_id)

    result = sy.VirtualWorker(sy.hook, worker_id, auto_add=auto_add)
    _objects = detail(worker, _objects)
    result._objects = _objects

    # make sure they weren't accidentally double registered
    for _, obj in _objects.items():
        if obj.id in worker._objects:
            del worker._objects[obj.id]

    return result


def _simplify_script_module(obj: torch.jit.ScriptModule) -> str:
    """Strategy to serialize a script module using Torch.jit"""
    return obj.save_to_buffer()


def _detail_script_module(worker: AbstractWorker, script_module_bin: str) -> torch.jit.ScriptModule:
    """"Strategy to deserialize a binary input using Torch load"""
    script_module_stream = io.BytesIO(script_module_bin)
    loaded_module = torch.jit.load(script_module_stream)
    return loaded_module


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

forced_full_simplifiers = {VirtualWorker: [17, _force_fullsimplify]}


def detail(worker: AbstractWorker, obj: object) -> object:
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
