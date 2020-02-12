"""
This file exists to provide one common place for all msgpack serialization to occur.
As msgpack only supports basic types and binary formats every type must be first be
converted to one of these types. Thus, we've split our functionality into three steps.
When converting from a PySyft object (or collection of objects) to an object to be
sent over the wire (a message), those three steps are (in order):

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
from typing import Callable

import inspect
import msgpack as msgpack_lib

import syft
from syft import dependency_check

from syft.federated.train_config import TrainConfig
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.private import PrivateTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.crt_precision import CRTPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.frameworks.torch.tensors.interpreters.placeholder import PlaceHolder
from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.pointers.pointer_plan import PointerPlan
from syft.generic.pointers.pointer_protocol import PointerProtocol
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.generic.string import String
from syft.execution.plan import Plan
from syft.execution.state import State
from syft.execution.protocol import Protocol
from syft.messaging.message import Message
from syft.messaging.message import OperationMessage
from syft.messaging.message import ObjectMessage
from syft.messaging.message import ObjectRequestMessage
from syft.messaging.message import IsNoneMessage
from syft.messaging.message import GetShapeMessage
from syft.messaging.message import ForceObjectDeleteMessage
from syft.messaging.message import SearchMessage
from syft.messaging.message import PlanCommandMessage
from syft.messaging.message import ExecuteWorkerFunctionMessage
from syft.serde import compression
from syft.serde.msgpack.native_serde import MAP_NATIVE_SIMPLIFIERS_AND_DETAILERS
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker

from syft.exceptions import GetNotPermittedError
from syft.exceptions import ResponseSignatureError

from syft.frameworks.torch.tensors.interpreters.gradients_core import GradFunc

if dependency_check.torch_available:
    from syft.serde.msgpack.torch_serde import MAP_TORCH_SIMPLIFIERS_AND_DETAILERS
else:
    MAP_TORCH_SIMPLIFIERS_AND_DETAILERS = {}

if dependency_check.tensorflow_available:
    from syft_tensorflow.serde import MAP_TF_SIMPLIFIERS_AND_DETAILERS
else:
    MAP_TF_SIMPLIFIERS_AND_DETAILERS = {}

from syft.serde.msgpack.proto import proto_type_info

# Maps a type to a tuple containing its simplifier and detailer function
# NOTE: serialization constants for these objects need to be defined in `proto.json` file
# in https://github.com/OpenMined/proto
MAP_TO_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    list(MAP_NATIVE_SIMPLIFIERS_AND_DETAILERS.items())
    + list(MAP_TORCH_SIMPLIFIERS_AND_DETAILERS.items())
    + list(MAP_TF_SIMPLIFIERS_AND_DETAILERS.items())
)

# If an object implements its own simplify and detail functions it should be stored in this list
# NOTE: serialization constants for these objects need to be defined in `proto.json` file
# in https://github.com/OpenMined/proto
OBJ_SIMPLIFIER_AND_DETAILERS = [
    AdditiveSharingTensor,
    FixedPrecisionTensor,
    PrivateTensor,
    CRTPrecisionTensor,
    LoggingTensor,
    MultiPointerTensor,
    PlaceHolder,
    ObjectPointer,
    Plan,
    State,
    Protocol,
    PointerTensor,
    PointerPlan,
    PointerProtocol,
    ObjectWrapper,
    TrainConfig,
    BaseWorker,
    AutogradTensor,
    Message,
    OperationMessage,
    ObjectMessage,
    ObjectRequestMessage,
    IsNoneMessage,
    GetShapeMessage,
    ForceObjectDeleteMessage,
    SearchMessage,
    PlanCommandMessage,
    GradFunc,
    String,
    ExecuteWorkerFunctionMessage,
]

# If an object implements its own force_simplify and force_detail functions it should be stored in this list
# NOTE: serialization constants for these objects need to be defined in `proto.json` file
# in https://github.com/OpenMined/proto
OBJ_FORCE_FULL_SIMPLIFIER_AND_DETAILERS = [BaseWorker]

# For registering syft objects with custom simplify and detail methods
# NOTE: serialization constants for these objects need to be defined in `proto.json` file
# in https://github.com/OpenMined/proto
EXCEPTION_SIMPLIFIER_AND_DETAILERS = [GetNotPermittedError, ResponseSignatureError]

## SECTION: High Level Simplification Router
def _force_full_simplify(worker: AbstractWorker, obj: object) -> object:
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
            forced_full_simplifiers[current_type][1](worker, obj),
        )
        return result
    # If we already tried to find a full simplifier for this type but failed, we should
    # simplify it instead.
    elif current_type in no_full_simplifiers_found:
        return _simplify(worker, obj)
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
                    forced_full_simplifiers[current_type][1](worker, obj),
                )
                return result

        # If there is not a full_simplifier for this
        # object, then we simplify it.
        no_full_simplifiers_found.add(current_type)
        return _simplify(worker, obj)


## SECTION: dinamically generate simplifiers and detailers
def _generate_simplifiers_and_detailers():
    """Generate simplifiers, forced full simplifiers and detailers,
    by registering native and torch types, syft objects with custom
    simplify and detail methods, or syft objects with custom
    force_simplify and force_detail methods.

    NOTE: this function uses `proto_type_info` that translates python class into Serde constant defined in
    https://github.com/OpenMined/proto. If the class used in `MAP_TO_SIMPLIFIERS_AND_DETAILERS`,
    `OBJ_SIMPLIFIER_AND_DETAILERS`, `EXCEPTION_SIMPLIFIER_AND_DETAILERS`, `OBJ_FORCE_FULL_SIMPLIFIER_AND_DETAILERS`
    is not defined in `proto.json` file in https://github.com/OpenMined/proto, this function will error.
    Returns:
        The simplifiers, forced_full_simplifiers, detailers
    """
    simplifiers = OrderedDict()
    forced_full_simplifiers = OrderedDict()
    detailers = OrderedDict()

    def _add_simplifier_and_detailer(curr_type, simplifier, detailer, forced=False):
        type_info = proto_type_info(curr_type)
        if forced:
            forced_full_simplifiers[curr_type] = (type_info.forced_code, simplifier)
            detailers[type_info.forced_code] = detailer
        else:
            simplifiers[curr_type] = (type_info.code, simplifier)
            detailers[type_info.code] = detailer

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
# Store types that use simplifiers from their ancestors so we
# can look them up quickly during serialization.
inherited_simplifiers_found = OrderedDict()


def _serialize_msgpack_simple(
    obj: object,
    worker: AbstractWorker = None,
    simplified: bool = False,
    force_full_simplification: bool = False,
) -> bin:
    if worker is None:
        # TODO[jvmancuso]: This might be worth a standalone function.
        worker = syft.framework.hook.local_worker

    # 1) Simplify
    # simplify difficult-to-serialize objects. See the _simpliy method
    # for details on how this works. The general purpose is to handle types
    # which the fast serializer cannot handle
    if not simplified:
        if force_full_simplification:
            simple_objects = _force_full_simplify(worker, obj)
        else:
            simple_objects = _simplify(worker, obj)
    else:
        simple_objects = obj

    return simple_objects


def _serialize_msgpack_binary(
    simple_objects: object,
    worker: AbstractWorker = None,
    simplified: bool = False,
    force_full_simplification: bool = False,
) -> bin:
    # 2) Serialize
    # serialize into a binary
    binary = msgpack_lib.dumps(simple_objects)

    # 3) Compress
    # compress the binary and return the result
    # prepend a 1-byte header '0' or '1' to the output stream
    # to denote whether output stream is compressed or not
    # if compressed stream length is greater than input stream
    # we output the input stream as it is with header set to '0'
    # otherwise we output the compressed stream with header set to '1'
    # even if compressed flag is set to false by the caller we
    # output the input stream as it is with header set to '0'
    return compression._compress(binary)


def serialize(
    obj: object,
    worker: AbstractWorker = None,
    simplified: bool = False,
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
        force_full_simplification (bool): Some objects are only partially serialized
            by default. For objects where this is the case, setting this flag to True
            will force the entire object to be serialized. For example, setting this
            flag to True will cause a VirtualWorker to be serialized WITH all of its
            tensors while by default VirtualWorker objects only serialize a small
            amount of metadata.

    Returns:
        binary: the serialized form of the object.
    """
    if worker is None:
        # TODO[jvmancuso]: This might be worth a standalone function.
        worker = syft.framework.hook.local_worker

    simple_objects = _serialize_msgpack_simple(obj, worker, simplified, force_full_simplification)
    return _serialize_msgpack_binary(simple_objects)


def _deserialize_msgpack_binary(binary: bin, worker: AbstractWorker = None) -> object:
    if worker is None:
        # TODO[jvmancuso]: This might be worth a standalone function.
        worker = syft.framework.hook.local_worker

    # 1) Decompress the binary if needed
    binary = compression._decompress(binary)

    # 2) Deserialize
    # This function converts the binary into the appropriate python
    # object (or nested dict/collection of python objects)
    simple_objects = msgpack_lib.loads(binary, use_list=False)

    # sometimes we want to skip detailing (such as in Plan)
    return simple_objects


def _deserialize_msgpack_simple(simple_objects: object, worker: AbstractWorker = None) -> object:
    if worker is None:
        # TODO[jvmancuso]: This might be worth a standalone function.
        worker = syft.framework.hook.local_worker

    # 3) Detail
    # This function converts typed, simple objects into their morefrom typing import Dict
    # complex (and difficult to serialize) counterparts which the
    # serialization library wasn't natively able to serialize (such
    # as msgpack's inability to serialize torch tensors or ... or
    # python slice objects
    return _detail(worker, simple_objects)


def deserialize(binary: bin, worker: AbstractWorker = None) -> object:
    if worker is None:
        # TODO[jvmancuso]: This might be worth a standalone function.
        worker = syft.framework.hook.local_worker

    simple_objects = _deserialize_msgpack_binary(binary, worker)
    return _deserialize_msgpack_simple(simple_objects, worker)


def _simplify(worker: AbstractWorker, obj: object, **kwargs) -> object:
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
    # breakpoint()
    current_type = type(obj)
    # print(current_type, current_type in simplifiers)
    if current_type in simplifiers:
        result = (simplifiers[current_type][0], simplifiers[current_type][1](worker, obj, **kwargs))
        return result
    elif current_type in inherited_simplifiers_found:
        result = (
            inherited_simplifiers_found[current_type][0],
            inherited_simplifiers_found[current_type][1](worker, obj, **kwargs),
        )
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
                inherited_simplifiers_found[current_type] = simplifiers[inheritance_type]
                result = (
                    inherited_simplifiers_found[current_type][0],
                    inherited_simplifiers_found[current_type][1](worker, obj, **kwargs),
                )
                return result

        # if there is not a simplifier for this
        # object, then the object is already a
        # simple python object and we can just
        # return it.
        no_simplifiers_found.add(current_type)
        return obj


def _detail(worker: AbstractWorker, obj: object, **kwargs) -> object:
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
        return detailers[obj[0]](worker, obj[1], **kwargs)
    else:
        return obj
