# external lib imports
import inspect

# external class/method imports (sorted by length)
from enum import Enum
from dataclasses import dataclass
from google.protobuf import json_format
from google.protobuf.message import Message
from typing import Union, Set

# syft import
from ..lazy_structures import LazySet, LazyDict
from ....decorators import syft_decorator
from ....proto.util.json_message_pb2 import JsonMessage


class Serializable:
    """When we want a custom object to be serializable within the Syft ecosystem
    (as outline in the tutorial above), the first thing we need to do is have it
    subclass from this class. You must then do the following in order for the
    subclass to be properly implemented:

    - implement a protobuf file in the "PySyft/proto" folder for this custom class.
    - compile the protobuf file by running `bash scripts/build_proto`
    - find the generated python file in syft.proto
    - import the generated protobuf class into my custom class
    - implement get_protobuf_Schema
    - implement <my class>._object2proto() method to serialize the object to protobuf
    - implement <my class>._proto2object() to deserialize the protobuf object

    At this point, your class should be ready to serialize and deserialize! Don't
    forget to add tests for your object!

    If you want to wrap an existing type (like a torch.tensor) to be used in our serialization
    ecosystem, you should consider wrapping it. Wrapping means that we NEVER use the wrapper
    further more into our ecosystem, we only need an easy interface to serialize wrappers.

    Eg:

    class WrapperInt:
        def __init__(self, int_obj: int):
            self.int_obj = int_obj

        def _object2proto(self) -> WrapperIntPB:
            ...

        @staticmethod
        def _proto2object(proto) -> int:
            ...

        @staticmethod
        def get_protobuf_schema() -> type:
            ...

        @staticmethod
        def get_wrapped_type() -> type:
            return int


    You must implement the following in order for the subclass to be properly implemented to be
    seen as a wrapper:
    - everything presented in the first tutorial of this docstring.
    - implement get_wrapped_type to return the wrapped type.

    Note: A wrapper should NEVER be used in the codebase, these are only for serialization purposes
    on external objects.

    After doing all of the above steps, you can call something like sy.serialize(5) and be
    serialized using our messaging proto backbone.
    """

    @staticmethod
    def _proto2object(proto: Message) -> "Serializable":
        """This method converts a protobuf object into a subclass of Serializable

        This method must be implemented for all classes which subclassSerializable - namely
        all classes which can be serialized within the Syft ecosystem. It should convert the
        corresponding protobuf message for the subclass into an instance of the class. This
        allows all the logic which goes from protobuf message to other formats (JSON, binary, etc.)
        to be generic and simply inherited from the rest of this class.

        :param proto: the protobuf object to be converted into an instance of type(self)
        :param type: Message
        :return: an instance of type(self)
        :rtype: Serializable

        """
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Message:
        """This methods converts self into a protobuf object

        This method must be implemented by all subclasses so that generic high-level functions
        implemented here (such as .json(), .binary(), etc) know how to convert the object into
        a protobuf object before further converting it into the requested format.

        :return: a protobuf message
        :rtype: Message
        """

        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema() -> type:
        """
            This static method returns the schema used when serializing this
            class.

            :return: a protobuf type message
            :rtype: type

        """
        raise NotImplementedError

    @staticmethod
    def get_wrapped_type():
        """
            This static method returns the wrapped type, if the current class is
            a wrapper over an external object.

            :return: the wrapped type
            :rtype: type
        """
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def to_proto(self) -> Message:
        """A convenience method to convert any subclass of Serializable into a protobuf object.

        :return: a protobuf message
        :rtype: Message
        """
        return self.serialize(to_proto=True)

    @syft_decorator(typechecking=True)
    def proto(self) -> Message:
        """A convenience method to convert any subclass of Serializable into a protobuf object.

        :return: a protobuf message
        :rtype: Message
        """
        return self.serialize(to_proto=True)

    @syft_decorator(typechecking=True)
    def to_json(self) -> str:
        """A convenience method to convert any subclass of Serializable into a JSON object.

        :return: a JSON string
        :rtype: str
        """
        return self.serialize(to_json=True)

    @syft_decorator(typechecking=True)
    def json(self) -> str:
        """A convenience method to convert any subclass of Serializable into a JSON object.

        :return: a JSON string
        :rtype: str
        """
        return self.serialize(to_json=True)

    @syft_decorator(typechecking=True)
    def to_binary(self) -> bytes:
        """A convenience method to convert any subclass of Serializable into a binary object.

        :return: a binary string
        :rtype: bytes
        """
        return self.serialize(to_binary=True)

    @syft_decorator(typechecking=True)
    def binary(self) -> bytes:
        """A convenience method to convert any subclass of Serializable into a binary object.

        :return: a binary string
        :rtype: bytes
        """
        return self.serialize(to_binary=True)

    @syft_decorator(typechecking=True)
    def to_hex(self) -> str:
        """A convenience method to convert any subclass of Serializable into a hex object.

        :return: a hex string
        :rtype: str
        """
        return self.serialize(to_hex=True)

    @syft_decorator(typechecking=True)
    def hex(self) -> str:
        """A convenience method to convert any subclass of Serializable into a hex object.

        :return: a hex string
        :rtype: str
        """
        return self.serialize(to_hex=True)

    @syft_decorator(typechecking=True)
    def serialize(
        self,
        to_proto: bool = True,
        to_json: bool = False,
        to_binary: bool = False,
        to_hex: bool = False,
    ) -> Union[str, bytes, Message]:
        """Serialize the object according to the parameters.

        This is the primary serialization method, which processes the above
        flags in a particular order. In general, it is not expected that people
        will set multiple to_<type> flags to True at the same time. We don't
        currently have logic which prevents this, because this may affect
        runtime performance, but if several flags are True, then we will simply
        take return the type of latest supported flag from the following list:

            - proto
            - json
            - binary
            - hex

        TODO: we could also add "dict" to this list but it's not clear if it would be used.

        :param to_proto: set this flag to TRUE if you want to return a protobuf object
        :type to_proto: bool
        :param to_json: set this flag to TRUE if you want to return a json object
        :type to_json: bool
        :param to_binary: set this flag to TRUE if you want to return a binary object
        :type to_binary: bool
        :param to_hex: set this flag to TRUE if you want to return a hex string object
        :type to_hex: bool
        :return: a serialized form of the object on which serialize() is called.
        :rtype: Union[str, bytes, Message]

        """

        if to_binary:
            return self._object2proto().SerializeToString()
        elif to_hex:
            return self._object2proto().SerializeToString().hex()
        elif to_json:
            blob = json_format.MessageToJson(message=self._object2proto())
            blob = json_format.MessageToJson(
                message=JsonMessage(obj_type=type(self).__qualname__, content=blob)
            )
            return blob
        elif to_proto:
            return type(self)._object2proto(self)
        else:
            raise Exception(
                """You must specify at least one deserialization format using
                            one of the arguments of the serialize() method such as:
                            to_proto, to_json, to_binary, or to_hex."""
            )


def get_protobuf(cls: type) -> (Set[Serializable], Set[Serializable]):
    """
        Function to retrieve all claases that implement the Serializable interface.

        These types can be either be a native type (that we implement and use in syft) or an
        external type (that we want to use like an external type, like just use it) but serialize it
        on our messaging backbone.

        A type that wants to implement serialization using syft hast to implement from Serializable:
            1. _object2proto
            2. proto2object
            3. get_protobuf_schema

        If these methods are not implemented, the class won't be enrolled in the
        set of types that can be serialized using syft.

        If you want to become a wrapper you have to implement as well:
            4. get_wrapped_type
    """

    class SerdeTypes(Enum):
        """
            Enum class to represent the possible results of the check_type
            function.

                1. NotSerdeType - not a serializable type, even if it
                implements Serializable.
                2. SerdeNativeType - a native syft type.
                3. SerdeWrapperType - a wrapped over an imported type.
        """

        NotSerdeType = 0
        SerdeNativeType = 1
        SerdeWrapperType = 2

    def check_type(s) -> SerdeTypes:
        """
            Check if a class has:
                1. bufferize implemented.
                2. unbufferize implemented.
                3. get_protobuf_schema implemented.
                4. no abstact methods.
                5. get_original_class method
            To be sure that it can be used with protobufers.
        """
        # checking if the class is not abstract
        not_abstract = not inspect.isabstract(s)

        # checking if the class is actually implementing _obj2proto and not
        # inheriting it from the parent class or skipping its implementation(
        # forbidden).
        object2proto_implemented = s._object2proto.__qualname__.startswith(s.__name__)

        # checking if the class is actually implementing _proto2obj and not
        # inheriting it from the parent class or skipping its implementation(
        # forbidded).
        proto2object_implemented = s._proto2object.__qualname__.startswith(s.__name__)

        # checking if the class is actually implementing get_schema and not
        # inheriting it from the parent class or skipping its implementation
        # (forbidded)
        get_schema_implemented = s.get_protobuf_schema.__qualname__.startswith(
            s.__name__
        )

        # checking if the class is actually implementing get_wrapped_type and
        # not inheriting it from the parent class or skipping its
        # implementation (forbidden).
        get_wrapped_type_implemented = s.get_wrapped_type.__qualname__.startswith(
            s.__name__
        )

        # this is a check to see if the must implement methods were implemented,
        # if not, we cannot serialize this type. Note: we might still be able
        # to serialize its children types, but we handle this in the main
        # function.
        valid_serde = (
            not_abstract
            and object2proto_implemented
            and proto2object_implemented
            and get_schema_implemented
        )

        # if its not a valid serde type, return NotSerdeType.
        if not valid_serde:
            return SerdeTypes.NotSerdeType

        # if it is a valid serde type and it implements the get_wrapped method,
        # return SerdeWrapperType.
        if get_wrapped_type_implemented:
            return SerdeTypes.SerdeWrapperType

        # if it is a valid type and it does implement the get_wrapped method, it
        # meants that is a native type, return SerdeNativeType
        return SerdeTypes.SerdeNativeType

    # the types that we implement and we want to make them serializable
    native_types = set()

    # the types that we import and we want to make them serializable
    wrapper_types = set()

    # get all subclasses of the current class, direct children.
    for s in cls.__subclasses__():
        # check what type of serde object we have
        serde_type = check_type(s)

        # check if the serde_type is a native type and add it if yes
        if serde_type is SerdeTypes.SerdeNativeType:
            native_types.add(s)

        # check if the serde_type is a wrapper type and add it if yes
        if serde_type is SerdeTypes.SerdeWrapperType:
            wrapper_types.add(s)

        # even if the current class is not serializable (might be and abstract
        # class or just isn't supposed to be serialized, the children could
        # be serializable, continue the tree search and add the results to the
        # native and wrapper sets.
        for c in s.__subclasses__():
            sub_native_set, sub_wrapper_set = get_protobuf(c)
            native_types.union(sub_native_set)
            wrapper_types.union(sub_wrapper_set)

    return native_types, wrapper_types


def update_serde_cache() -> None:
    """
        Function for the serde cache updates when it does not find a type. This
        means that the type is not serializable and and error will be thrown or
        the type was not found due to the fact that the cache is stale and we
        want to update all the types in the cache.
    """

    # there are two serializable classes, native and wrapper ones. You can read
    # more in get_protobuf about this
    native_types, wrapper_types = get_protobuf(Serializable)
    for native_type in native_types:
        # add the native type to the available types LazySet. If a type is not
        # in this set it means that it is unknown to the serde store.
        serde_store.available_types.add(native_type)

        # cache the mapping from the qualname to the actual type, this can be
        # done through pydoc.locate as well.
        serde_store.qual_name2type[native_type.__qualname__] = native_type

        # update the mappings from type to the actual protobuf schema and from
        # the protobuf to the type that it serializes for faster searches
        serde_store.type2schema[native_type] = native_type.get_protobuf_schema()
        serde_store.schema2type[native_type.get_protobuf_schema()] = native_type

    for native_wrapper in wrapper_types:
        # add the wrapper type and the wrapped type to the available types
        # LazySet. If a type is not in this set it means that it is unknown
        # to the serde store.
        serde_store.available_types.add(native_wrapper)
        serde_store.available_types.add(native_wrapper.get_wrapped_type())

        # add the wrapped type to the wrapped types LazySet
        serde_store.wrapped_types.add(native_wrapper.get_wrapped_type())

        # cache the mapping from the qualname to the actual native type,
        # this can be done through pydoc.locate as well.
        serde_store.qual_name2type[native_wrapper.__qualname__] = native_wrapper

        # update the mappings from wrapper type to the protobuf schema and the
        # other way around
        serde_store.type2schema[native_wrapper] = native_wrapper.get_protobuf_schema()
        serde_store.schema2type[native_wrapper.get_protobuf_schema()] = native_wrapper

        # update the mappings from wrapped type to the protobuf schema and the
        # other way around
        serde_store.wrapped2wrapper[native_wrapper.get_wrapped_type()] = native_wrapper
        serde_store.wrapper2wrapped[native_wrapper] = native_wrapper.get_wrapped_type()


@dataclass(frozen=True)
class SerdeStore:
    # set of all serializable types.
    available_types = LazySet(update_rule=update_serde_cache)
    # set of all wrapped types.
    wrapped_types = LazySet(update_rule=update_serde_cache)
    # mapping from the name of the type and actual type (will be removed)
    qual_name2type = LazyDict(update_rule=update_serde_cache)
    # mapping from the type to the schema that serializes that type
    type2schema = LazyDict(update_rule=update_serde_cache)
    # mapping from the schema to the type it serializes
    schema2type = LazyDict(update_rule=update_serde_cache)
    # mapping from the wrapped type to its wrapper. Eg: WrapperTorchTensor -> TorchTensor
    wrapped2wrapper = LazyDict(update_rule=update_serde_cache)
    # mapping from the wrapper type to its wrapped type. Eg: TorchTensor -> WrapperTorchTensor
    wrapper2wrapped = LazyDict(update_rule=update_serde_cache)


serde_store = SerdeStore()
