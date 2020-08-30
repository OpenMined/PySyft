from collections import OrderedDict

import inspect
import re
from dataclasses import dataclass

import syft
from syft.messaging.message import ObjectMessage
from syft.messaging.message import TensorCommandMessage
from syft.serde import compression
from syft.workers.abstract import AbstractWorker

from syft_proto.messaging.v1.message_pb2 import SyftMessage as SyftMessagePB
from syft_proto.types.syft.v1.arg_pb2 import Arg as ArgPB
from syft_proto.types.syft.v1.arg_pb2 import ArgList as ArgListPB
from syft.generic.abstract.syft_serializable import (
    SyftSerializable,
    get_protobuf_classes,
    get_protobuf_wrappers,
)


class MetaProtobufGlobalState(type):
    """
    Metaclass that wraps all properties in ProtobufGlobalState to be updated
    when the global state is marked as stale.
    """

    @staticmethod
    def wrapper(wrapped_func: property) -> property:
        """
        Method to generate the new property.

        Args:
            wrapped_func (Property): property of the generated type.

        Returns:
             Property: new property that is wrapped to get updated when the global state
             is marked as stale.
        """

        @property
        def wrapper(self):
            """
            Generated new property that forces updates if the global state is marked as stale.
            """
            self = self.update()
            return wrapped_func.__get__(self, type(self))

        return wrapper

    def __new__(meta, classname, bases, class_dict):
        """
        Method to generate the new type, wrapping all properties in the given type.
        """
        for attr_name, attr_body in class_dict.items():
            if isinstance(attr_body, property):
                class_dict[attr_name] = MetaProtobufGlobalState.wrapper(attr_body)
        return type.__new__(meta, classname, bases, class_dict)


@dataclass
class ProtobufGlobalState(metaclass=MetaProtobufGlobalState):
    """
    Class to generate a global state of the protobufers in a lazy way. All attributes
    should be used by their properties, not by their hidden value.

    The global state can be marked as stale by setting stale_state to False, forcing
    the next usage of the to be updated, enabling dynamic types in serde.

    All types should be enrolled in proto.json in syft-serde (soon to be deprecated,
    when msgpack is removed).

    Attributes:

        _OBJ_FORCE_FULL_PROTOBUF_TRANSLATORS (list): If a type implements its own
        force_bufferize and force_unbufferize functions, it should be stored in this list.
        This will become deprecated soon.

        _bufferizers (OrderedDict): The mapping from a type to its own bufferizer.

        _forced_full_bufferizers (OrderedDict): The mapping from a type to its own forced
        bufferizer.

        _unbufferizers (OrderedDict): The mapping from a type to its own unbufferizer.

        _no_bufferizers_found (set): In this set we store the primitives that we cannot
        bufferize anymore.

        _no_full_bufferizers_found (set): In this set we store the primitives that we cannot
        force bufferize anymore.

        _inherited_bufferizers_found (OrderedDict): In this dict we store the any inherited
        bufferizer that a type can use. This might become deprecated

        stale_state (Bool): Marks the global state to be stale or not.
    """

    _OBJ_FORCE_FULL_PROTOBUF_TRANSLATORS = []
    _bufferizers = OrderedDict()
    _forced_full_bufferizers = OrderedDict()
    _unbufferizers = OrderedDict()
    _no_bufferizers_found = set()
    _no_full_bufferizers_found = set()
    _inherited_bufferizers_found = OrderedDict()

    stale_state = True

    @property
    def obj_force_full_protobuf_translators(self):
        return self._OBJ_FORCE_FULL_PROTOBUF_TRANSLATORS

    @property
    def forced_full_bufferizers(self):
        return self._forced_full_bufferizers

    @property
    def bufferizers(self):
        return self._bufferizers

    @property
    def unbufferizers(self):
        return self._unbufferizers

    @property
    def no_bufferizers_found(self):
        return self._no_bufferizers_found

    @property
    def no_full_bufferizers_found(self):
        return self._no_full_bufferizers_found

    @property
    def inherited_bufferizers_found(self):
        return self._inherited_bufferizers_found

    def update(self):
        """
        Updates the global state of protobuf.
        """
        if not self.stale_state:
            return self

        obj_protobuf_translators = list(get_protobuf_classes(SyftSerializable))
        obj_protobuf_wrappers = list(get_protobuf_wrappers(SyftSerializable))

        def _add_bufferizer_and_unbufferizer(
            curr_type, proto_type, bufferizer, unbufferizer, forced=False
        ):

            if forced:
                self._forced_full_bufferizers[curr_type] = bufferizer
                self._unbufferizers[proto_type] = unbufferizer
            else:
                self._bufferizers[curr_type] = bufferizer
                self._unbufferizers[proto_type] = unbufferizer

        for curr_type in obj_protobuf_translators:
            _add_bufferizer_and_unbufferizer(
                curr_type,
                curr_type.get_protobuf_schema(),
                curr_type.bufferize,
                curr_type.unbufferize,
            )

        for curr_type in obj_protobuf_wrappers:
            _add_bufferizer_and_unbufferizer(
                curr_type.get_original_class(),
                curr_type.get_protobuf_schema(),
                curr_type.bufferize,
                curr_type.unbufferize,
            )

        for syft_type in self._OBJ_FORCE_FULL_PROTOBUF_TRANSLATORS:
            proto_type = syft_type.get_protobuf_schema()
            force_bufferizer, force_unbufferizer = (
                syft_type.force_bufferize,
                syft_type.force_unbufferize,
            )
            _add_bufferizer_and_unbufferizer(
                syft_type, proto_type, force_bufferizer, force_unbufferizer, forced=True
            )

        self.stale_state = False
        return self


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

    current_type = type(obj)
    if current_type in protobuf_global_state.bufferizers:
        result = protobuf_global_state.bufferizers[current_type](worker, obj, **kwargs)
        return result
    elif current_type in protobuf_global_state.inherited_bufferizers_found:
        return (
            protobuf_global_state.inherited_bufferizers_found[current_type][0],
            protobuf_global_state.inherited_bufferizers_found[current_type][1](
                worker, obj, **kwargs
            ),
        )
    # If we already tried to find a bufferizer for this type but failed, we should
    # just return the object as it is.
    elif current_type in protobuf_global_state.no_bufferizers_found:
        raise Exception(f"No corresponding Protobuf message found for {current_type}")
    else:

        protobuf_global_state.stale_state = True
        if current_type in protobuf_global_state.bufferizers:
            result = protobuf_global_state.bufferizers[current_type](worker, obj, **kwargs)
            return result

        # If the object type is not in bufferizers,
        # we check the classes that this object inherits from.
        # `inspect.getmro` give us all types this object inherits
        # from, including `type(obj)`. We can skip the type of the
        # object because we already tried this in the
        # previous step.
        classes_inheritance = inspect.getmro(type(obj))[1:]

        for inheritance_type in classes_inheritance:
            if inheritance_type in protobuf_global_state.bufferizers:
                # Store the inheritance_type in bufferizers so next time we see this type
                # serde will be faster.
                protobuf_global_state.inherited_bufferizers_found[
                    current_type
                ] = protobuf_global_state.bufferizers[inheritance_type]
                result = protobuf_global_state.inherited_bufferizers_found[current_type](
                    worker, obj, **kwargs
                )
                return result

        protobuf_global_state.no_bufferizers_found.add(current_type)
        raise Exception(f"No corresponding Protobuf message found for {current_type}")


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
    if current_type in protobuf_global_state.forced_full_bufferizers:
        result = (
            protobuf_global_state.forced_full_bufferizers[current_type][0],
            protobuf_global_state.forced_full_bufferizers[current_type][1](worker, obj),
        )
        return result
    # If we already tried to find a full bufferizer for this type but failed, we should
    # bufferize it instead.
    elif current_type in protobuf_global_state.no_full_bufferizers_found:
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
            if inheritance_type in protobuf_global_state.forced_full_bufferizers:
                # Store the inheritance_type in forced_full_bufferizers so next
                # time we see this type serde will be faster.
                protobuf_global_state.forced_full_bufferizers[
                    current_type
                ] = protobuf_global_state.forced_full_bufferizers[inheritance_type]
                result = (
                    protobuf_global_state.forced_full_bufferizers[current_type][0],
                    protobuf_global_state.forced_full_bufferizers[current_type][1](worker, obj),
                )
                return result

        # If there is not a full_bufferizer for this
        # object, then we bufferize it.
        protobuf_global_state.no_full_bufferizers_found.add(current_type)
        return _bufferize(worker, obj)


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
    if isinstance(obj_type, None):
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
    """This method can deserialize any object PySyft needs to send or store.

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
    current_type = type(obj)
    if current_type in protobuf_global_state.unbufferizers:
        return protobuf_global_state.unbufferizers[current_type](worker, obj, **kwargs)
    else:
        protobuf_global_state.stale_state = True
        if current_type in protobuf_global_state.unbufferizers:
            result = protobuf_global_state.bufferizers[current_type](worker, obj, **kwargs)
            return result

        raise Exception(f"No unbufferizer found for {current_type}")


def bufferize_args(worker: AbstractWorker, args_: list) -> list:
    return [bufferize_arg(worker, arg) for arg in args_]


def bufferize_arg(worker: AbstractWorker, arg: object) -> ArgPB:
    protobuf_arg = ArgPB()

    if isinstance(arg, list):
        protobuf_arg_list = ArgListPB()
        arg_list = [bufferize_arg(worker, i) for i in arg]
        protobuf_arg_list.args.extend(arg_list)
        protobuf_arg.arg_list.CopyFrom(protobuf_arg_list)

    else:
        attr_name = "arg_" + _camel2snake(type(arg).__name__)

        try:
            setattr(protobuf_arg, attr_name, arg)
        except:
            getattr(protobuf_arg, attr_name).CopyFrom(_bufferize(worker, arg))

    return protobuf_arg


def unbufferize_args(worker: AbstractWorker, protobuf_args: list) -> list:
    return tuple((unbufferize_arg(worker, arg) for arg in protobuf_args))


def unbufferize_arg(worker: AbstractWorker, protobuf_arg: ArgPB) -> object:
    protobuf_field_name = protobuf_arg.WhichOneof("arg")

    protobuf_arg_field = getattr(protobuf_arg, protobuf_field_name)

    if protobuf_field_name == "arg_list":
        arg = [unbufferize_arg(worker, i) for i in protobuf_arg_field.args]
    else:
        try:
            arg = _unbufferize(worker, protobuf_arg_field)
        except:
            arg = protobuf_arg_field

    return arg


def _camel2snake(string: str):
    return string[0].lower() + re.sub(r"(?!^)[A-Z]", lambda x: "_" + x.group(0).lower(), string[1:])


protobuf_global_state = ProtobufGlobalState()
