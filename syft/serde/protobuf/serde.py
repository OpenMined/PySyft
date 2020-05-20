from collections import OrderedDict

import inspect
import re
import syft
from syft import dependency_check
from syft.messaging.message import ObjectMessage
from syft.messaging.message import TensorCommandMessage
from syft.serde import compression
from syft.serde.protobuf.native_serde import MAP_NATIVE_PROTOBUF_TRANSLATORS
from syft.workers.abstract import AbstractWorker

from syft_proto.messaging.v1.message_pb2 import SyftMessage as SyftMessagePB
from syft_proto.types.syft.v1.arg_pb2 import Arg as ArgPB
from syft.serde.syft_serializable import SyftSerializable, get_protobuf_subclasses

if dependency_check.torch_available:
    from syft.serde.protobuf.torch_serde import MAP_TORCH_PROTOBUF_TRANSLATORS
else:
    MAP_TORCH_PROTOBUF_TRANSLATORS = {}

# if dependency_check.tensorflow_available:
#     from syft_tensorflow.serde import MAP_TF_PROTOBUF_TRANSLATORS
# else:
#     MAP_TF_PROTOBUF_TRANSLATORS = {}

from syft.serde.protobuf.proto import MAP_PYTHON_TO_PROTOBUF_CLASSES

# Maps a type to its bufferizer and unbufferizer functions
MAP_TO_PROTOBUF_TRANSLATORS = OrderedDict(
    list(MAP_NATIVE_PROTOBUF_TRANSLATORS.items())
    + list(MAP_TORCH_PROTOBUF_TRANSLATORS.items())
    # + list(MAP_TF_PROTOBUF_TRANSLATORS.items())
)

# If an object implements its own bufferize and unbufferize functions it should be stored in this list
OBJ_PROTOBUF_TRANSLATORS = None

# If an object implements its own force_bufferize and force_unbufferize functions it should be stored in this list
# OBJ_FORCE_FULL_PROTOBUF_TRANSLATORS = [BaseWorker]
OBJ_FORCE_FULL_PROTOBUF_TRANSLATORS = []

# For registering syft objects with custom bufferize and unbufferize methods
# EXCEPTION_PROTOBUF_TRANSLATORS = [GetNotPermittedError, ResponseSignatureError]
EXCEPTION_PROTOBUF_TRANSLATORS = []


def get_bufferizers():
    """
        Function to retrieve the bufferizers, so that no other function uses directly de global elements.
    """
    init_global_vars()
    return bufferizers.items()


def init_global_vars():
    """
        Function to initialise at the first usage all the global elements used in protobuf/serde.py and protobuf/proto.py.
    """
    global OBJ_PROTOBUF_TRANSLATORS, bufferizers, forced_full_bufferizers, unbufferizers, MAP_PYTHON_TO_PROTOBUF_CLASSES
    if OBJ_PROTOBUF_TRANSLATORS is None:
        OBJ_PROTOBUF_TRANSLATORS = list(get_protobuf_subclasses(SyftSerializable))
        for proto_class in OBJ_PROTOBUF_TRANSLATORS:
            MAP_PYTHON_TO_PROTOBUF_CLASSES[proto_class] = proto_class.get_protobuf_schema()
        (
            bufferizers,
            forced_full_bufferizers,
            unbufferizers,
        ) = _generate_bufferizers_and_unbufferizers()


## SECTION: High Level Translation Router
def _force_full_bufferize(worker: AbstractWorker, obj: object) -> object:
    """To force a full bufferize conversion generally if the usual _bufferize is not suitable.

    If we can not full convert an object we convert it as usual instead.

    Args:
        obj: The object.

    Returns:
        The bufferize object.
    """
    # check to see if there is a full bufferize converter
    # for this type. If there is, return the full converted object.
    current_type = type(obj)
    if current_type in forced_full_bufferizers:
        result = (
            forced_full_bufferizers[current_type][0],
            forced_full_bufferizers[current_type][1](worker, obj),
        )
        return result
    # If we already tried to find a full bufferizer for this type but failed, we should
    # bufferize it instead.
    elif current_type in no_full_bufferizers_found:
        return _bufferize(worker, obj)
    else:
        # If the object type is not in forced_full_bufferizers,
        # we check the classes that this object inherits from.
        # `inspect.getmro` give us all types this object inherits
        # from, including `type(obj)`. We can skip the type of the
        # object because we already tried this in the
        # previous step.
        classes_inheritance = inspect.getmro(type(obj))[1:]

        for inheritance_type in classes_inheritance:
            if inheritance_type in forced_full_bufferizers:
                # Store the inheritance_type in forced_full_bufferizers so next
                # time we see this type serde will be faster.
                forced_full_bufferizers[current_type] = forced_full_bufferizers[inheritance_type]
                result = (
                    forced_full_bufferizers[current_type][0],
                    forced_full_bufferizers[current_type][1](worker, obj),
                )
                return result

        # If there is not a full_bufferizer for this
        # object, then we bufferize it.
        no_full_bufferizers_found.add(current_type)
        return _bufferize(worker, obj)


## SECTION: dinamically generate bufferizers and unbufferizers
def _generate_bufferizers_and_unbufferizers():
    """Generate bufferizers, forced full bufferizers and unbufferizers,
    by registering native and torch types, syft objects with custom
    bufferize and unbufferize methods, or syft objects with custom
    force_bufferize and force_unbufferize methods.

    NOTE: this function uses `proto_type_info` that translates python class into Serde constant defined in
    https://github.com/OpenMined/proto. If the class used in `MAP_TO_SIMPLIFIERS_AND_DETAILERS`,
    `OBJ_SIMPLIFIER_AND_DETAILERS`, `EXCEPTION_SIMPLIFIER_AND_DETAILERS`, `OBJ_FORCE_FULL_SIMPLIFIER_AND_DETAILERS`
    is not defined in `proto.json` file in https://github.com/OpenMined/proto, this function will error.
    Returns:
        The bufferizers, forced_full_bufferizers, unbufferizers
    """
    bufferizers = OrderedDict()
    forced_full_bufferizers = OrderedDict()
    unbufferizers = OrderedDict()

    def _add_bufferizer_and_unbufferizer(
        curr_type, proto_type, bufferizer, unbufferizer, forced=False
    ):

        if forced:
            forced_full_bufferizers[curr_type] = bufferizer
            unbufferizers[proto_type] = unbufferizer
        else:
            bufferizers[curr_type] = bufferizer
            unbufferizers[proto_type] = unbufferizer

    # Register native and torch types
    for curr_type in MAP_TO_PROTOBUF_TRANSLATORS:
        proto_type = MAP_PYTHON_TO_PROTOBUF_CLASSES[curr_type]
        bufferizer, unbufferizer = MAP_TO_PROTOBUF_TRANSLATORS[curr_type]
        _add_bufferizer_and_unbufferizer(curr_type, proto_type, bufferizer, unbufferizer)

    # Register syft objects with custom bufferize and unbufferize methods
    for syft_type in OBJ_PROTOBUF_TRANSLATORS + EXCEPTION_PROTOBUF_TRANSLATORS:
        proto_type = MAP_PYTHON_TO_PROTOBUF_CLASSES[syft_type]
        bufferizer, unbufferizer = syft_type.bufferize, syft_type.unbufferize
        _add_bufferizer_and_unbufferizer(syft_type, proto_type, bufferizer, unbufferizer)

    # Register syft objects with custom force_bufferize and force_unbufferize methods
    for syft_type in OBJ_FORCE_FULL_PROTOBUF_TRANSLATORS:
        proto_type = MAP_PYTHON_TO_PROTOBUF_CLASSES[syft_type]
        force_bufferizer, force_unbufferizer = (
            syft_type.force_bufferize,
            syft_type.force_unbufferize,
        )
        _add_bufferizer_and_unbufferizer(
            syft_type, proto_type, force_bufferizer, force_unbufferizer, forced=True
        )

    return bufferizers, forced_full_bufferizers, unbufferizers


bufferizers, forced_full_bufferizers, unbufferizers = None, None, None
# Store types that are not simplifiable (int, float, None) so we
# can ignore them during serialization.
no_bufferizers_found, no_full_bufferizers_found = set(), set()
# Store types that use simplifiers from their ancestors so we
# can look them up quickly during serialization.
inherited_bufferizers_found = OrderedDict()


## SECTION:  High Level Public Functions (these are the ones you use)
def serialize(
    obj: object,
    worker: AbstractWorker = None,
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

    init_global_vars()

    if worker is None:
        # TODO[jvmancuso]: This might be worth a standalone function.
        worker = syft.framework.hook.local_worker

    if force_no_serialization:
        # 0) Simplify
        # bufferize difficult-to-serialize objects. See the _bufferize method
        # for unbufferizes on how this works. The general purpose is to handle types
        # which the fast serializer cannot handle
        simple_objects = obj
        if not simplified:
            if force_full_simplification:
                simple_objects = _force_full_bufferize(worker, obj)
            else:
                simple_objects = _bufferize(worker, obj)
        return simple_objects

    # 1) Convert to Protobuf objects
    msg_wrapper = SyftMessagePB()

    protobuf_obj = _bufferize(worker, obj)

    obj_type = type(obj)
    if obj_type == type(None):
        msg_wrapper.contents_empty_msg.CopyFrom(protobuf_obj)
    elif obj_type == ObjectMessage:
        msg_wrapper.contents_object_msg.CopyFrom(protobuf_obj)
    elif obj_type == TensorCommandMessage:
        msg_wrapper.contents_action_msg.CopyFrom(protobuf_obj)

    # 2) Serialize
    # serialize into a binary
    binary = msg_wrapper.SerializeToString()

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
        return compression._compress(binary)


def deserialize(binary: bin, worker: AbstractWorker = None, unbufferizes=True) -> object:
    """ This method can deserialize any object PySyft needs to send or store.

    This is the high level function for deserializing any object or collection
    of objects which PySyft has sent over the wire or stored. It includes three
    steps, Decompress, Deserialize, and Detail as described inline below.

    Args:
        binary (bin): the serialized object to be deserialized.
        worker (AbstractWorker): the worker which is acquiring the message content,
            for example used to specify the owner of a tensor received(not obvious
            for virtual workers)
        unbufferizes (bool): there are some cases where we need to perform the decompression
            and deserialization part, but we don't need to unbufferize all the message.
            This is the case for Plan workers for instance

    Returns:
        object: the deserialized form of the binary input.
    """

    init_global_vars()

    if worker is None:
        # TODO[jvmancuso]: This might be worth a standalone function.
        worker = syft.framework.hook.local_worker

    # 1) Decompress the binary if needed
    binary = compression._decompress(binary)

    # 2) Deserialize
    msg_wrapper = SyftMessagePB()
    msg_wrapper.ParseFromString(binary)

    # 3) Convert back to a Python object
    message_type = msg_wrapper.WhichOneof("contents")
    python_obj = _unbufferize(worker, getattr(msg_wrapper, message_type))
    return python_obj


def _bufferize(worker: AbstractWorker, obj: object, **kwargs) -> object:
    """
    This function takes an object as input and returns a
    Protobuf object. The reason we have this function
    is that some objects are either NOT supported by high level (fast)
    serializers OR the high level serializers don't support the fastest
    form of serialization. For example, PyTorch tensors have custom pickle
    functionality thus its better to pre-serialize PyTorch tensors using
    pickle and then serialize the binary in with the rest of the message
    being sent.

    Args:
        obj: An object which needs to be converted to Protobuf.

    Returns:
        An Protobuf object which Protobuf can serialize.
    """

    # Check to see if there is a bufferizer
    # for this type. If there is, return the bufferized object.
    # breakpoint()
    init_global_vars()
    current_type = type(obj)
    if current_type in bufferizers:
        result = bufferizers[current_type](worker, obj, **kwargs)
        return result
    elif current_type in inherited_bufferizers_found:
        result = (
            inherited_bufferizers_found[current_type][0],
            inherited_bufferizers_found[current_type][1](worker, obj, **kwargs),
        )
        return result

    # If we already tried to find a bufferizer for this type but failed, we should
    # just return the object as it is.
    elif current_type in no_bufferizers_found:
        raise Exception(f"No corresponding Protobuf message found for {current_type}")

    else:
        # If the object type is not in bufferizers,
        # we check the classes that this object inherits from.
        # `inspect.getmro` give us all types this object inherits
        # from, including `type(obj)`. We can skip the type of the
        # object because we already tried this in the
        # previous step.
        classes_inheritance = inspect.getmro(type(obj))[1:]

        for inheritance_type in classes_inheritance:
            if inheritance_type in bufferizers:
                # Store the inheritance_type in bufferizers so next time we see this type
                # serde will be faster.
                inherited_bufferizers_found[current_type] = bufferizers[inheritance_type]
                result = inherited_bufferizers_found[current_type](worker, obj, **kwargs)
                return result

        no_bufferizers_found.add(current_type)
        raise Exception(f"No corresponding Protobuf message found for {current_type}")


def _unbufferize(worker: AbstractWorker, obj: object, **kwargs) -> object:
    """Reverses the functionality of _bufferize.
    Where applicable, it converts simple objects into more complex objects such
    as converting binary objects into torch tensors. Read _bufferize for more
    information on why _bufferize and unbufferize are needed.

    Args:
        worker: the worker which is acquiring the message content, for example
        used to specify the owner of a tensor received(not obvious for
        virtual workers).
        obj: a simple Python object which msgpack deserialized.

    Returns:
        obj: a more complex Python object which msgpack would have had trouble
            deserializing directly.
    """

    init_global_vars()

    current_type = type(obj)
    if current_type in unbufferizers:
        return unbufferizers[current_type](worker, obj, **kwargs)
    else:
        raise Exception(f"No unbufferizer found for {current_type}")


def bufferize_args(worker: AbstractWorker, args_: list) -> list:
    return [bufferize_arg(worker, arg) for arg in args_]


def bufferize_arg(worker: AbstractWorker, arg: object) -> ArgPB:
    protobuf_arg = ArgPB()

    attr_name = "arg_" + _camel2snake(type(arg).__name__)

    try:
        setattr(protobuf_arg, attr_name, arg)
    except:
        getattr(protobuf_arg, attr_name).CopyFrom(_bufferize(worker, arg))

    return protobuf_arg


def unbufferize_args(worker: AbstractWorker, protobuf_args: list) -> list:
    return tuple([unbufferize_arg(worker, arg) for arg in protobuf_args])


def unbufferize_arg(worker: AbstractWorker, protobuf_arg: ArgPB) -> object:
    protobuf_field_name = protobuf_arg.WhichOneof("arg")

    protobuf_arg_field = getattr(protobuf_arg, protobuf_field_name)
    try:
        arg = _unbufferize(worker, protobuf_arg_field)
    except:
        arg = protobuf_arg_field

    return arg


def _camel2snake(string: str):
    return string[0].lower() + re.sub(r"(?!^)[A-Z]", lambda x: "_" + x.group(0).lower(), string[1:])
