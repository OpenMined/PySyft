"""

***************************************************
Tutorial: Serialization and Deserialization in Syft
***************************************************

In this file, we have the main Serializable class which orchestrates
the serialization of objects within the Syft ecosystem. Users and
developers of Syft need to serialize objects for a myriad of reasons,
but the most common 3 are:

- to save an object into a database which requires serialization (such as key-value dbs like Redis)
- to send an object over the network (any protocol).
- to save an object to disk.

All serialization in Syft uses a library called 'protobuf'.
This is a technology developed at Google for fast, secure
serialization of objects (https://developers.google.com/protocol-buffers).
We use an existing library like Protobuf for multiple reasons:

- Protobuf creates cross-language serialization abilities
- Protobuf is secure
- Protobuf is fast.

In short, lots of projects need serialization and so a lot of time
and effort has gone into creating great serialization libraries. Thus,
we want to inherit the work of others instead of having to reinvest the time
to build these things ourselves.

.. note:: DO NOT attempt to create your own serialization by
    creating strings out of objects yourself. Not only will this almost
    certainly be slower than using a technology like protobuf,
    but you will require everyone to re-implement your serialization
    techniques in every language which needs to support your object
    AND you run the risk of introducing dangerous SECURITY FLAWS.
    This is a place where we definitely want to use a robust library
    like protobuf.


Serializing and Deserializing Syft Objects:
###########################################

If you want to serialize an object in the syft ecosystem, the easiest way
to do so is to just call one of the serialization methods which we install
on the object for your convenience::

    import syft as sy

    # ObjectWithID is the simplest object in the Syft ecosystem
    # It's the parent class of many other classes. As the name
    # implies, it's literally just an object with an ID.
    # We'll use it to show serialization, but you can use the
    # same approach with any serializable object in the Syft
    # ecosystem.
    from syft.core.common.object import ObjectWithID

    # this creates an object which has an id
    my_object = ObjectWithID()

    print(my_object)
    # >>> <ObjectWithID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>

    # by default, .serialize() will serialize it to a protobuf Message object
    proto_obj = my_object.serialize()

    print(proto_obj)
    # >>> obj_type: "syft.core.common.object.ObjectWithID"
    # >>> id {
    # >>>   obj_type: "syft.core.common.uid.UID"
    # >>>   value: "23hi23hgo2ih23ih2;o3igh;2oih;iagapwihpag"
    # >>> }

    # deserialization also assumes you are accepting a protobuf object
    my_object_again = sy.deserialize(blob=proto_obj)

    print(my_object_again)
    # >>> <ObjectWithID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>

However, perhaps the best thing about protobuf is that it can easily
be turned into a wide variety of very portable representations. We have
convenience functions for 4 popular represenations: protobuf, json, binary,
and hex. Setup::

    import syft as sy
    from syft.core.common.object import ObjectWithID

    obj = ObjectWithId()

Protobuf
********

.. code::

    proto_obj = obj.serialize(to_proto=True)
    proto_obj = obj.to_proto()
    proto_obj = obj.proto()

    print(proto_obj)
    # >>> obj_type: "syft.core.common.object.ObjectWithID"
    # >>> id {
    # >>>   obj_type: "syft.core.common.uid.UID"
    # >>>   value: "23hi23hgo2ih23ih2;o3igh;2oih;iagapwihpag"
    # >>> }

    obj_again = sy.deserialize(blob=proto_obj, from_proto=True)

JSON
****

.. code::

    json_obj = obj.serialize(to_json=True)
    json_obj = obj.to_json()
    json_obj = obj.json()

    print(json_obj)
    # >>> {
    # >>>   "objType": "syft.core.common.object.ObjectWithID",
    # >>>   "id": {
    # >>>     "objType": "syft.core.common.uid.UID",
    # >>>     "value": "+xuwZ1u3TEm+zucAqwoVFA=="
    # >>>   }
    # >>> }

    obj_again = sy.deserialize(blob=proto_obj, from_json=True)

Binary
******

.. code::

    binary_obj = obj.serialize(to_binary=True)
    binary_obj = obj.to_binary()
    binary_obj = obj.binary()

    # print(binary_obj)
    # >>> b'{  "objType": "syft.core.common.object.ObjectWithID",
    # >>> "id": {    "objType": "syft.core.common.uid.UID",
    # >>> "value": "+xuwZ1u3TEm+zucAqwoVFA=="  }}'

    obj_again = sy.deserialize(blob=proto_obj, from_binary=True)

Now we can continue with the class definition for the Serializable class, which
is the parent class for all serializable objects within Syft.

If you'd like to see a simple example of a class which can be serialized, please read
the source code of :py:mod:`syft.core.common.object.ObjectWithID`.
"""

# external lib imports
import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable
import inspect

# external class/method imports (sorted by length)
from google.protobuf.message import Message
from google.protobuf import json_format

from syft.core.common.lazy_structures import LazyDict, LazySet

def get_protobuf(cls: type) -> (set, set):
    """
        Function to retrieve all wrappers that implement the protobuf methods from the
        SyftSerializable class:
        A type that wants to implement to wrap another type (eg. torch.Tensor) for the protobuf
        interface and to use it with syft-proto has to inherit SyftSerializable (directly or
        from the parent class) and to implement
        (cannot inherit from parent class):
            1. bufferize
            2. unbufferize
            3. get_protobuf_schema
            4. get_original_class
        If these methods are not implemented, the class won't be enrolled in the types that
        are wrappers can't use syft-proto.
    """
    class SerdeTypes(Enum):
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

        # checking if the class is actually implementing _obj2proto and not inheriting it from
        # the parent class or skipping its implementation(forbidden).
        obj2proto_implemented = s._obj2proto.__qualname__.startswith(s.__name__)

        # checking if the class is actually implementing _proto2obj and not inheriting it from
        # the parent class or skipping its implementation(forbidded).
        proto2obj_implemented = s._proto2obj.__qualname__.startswith(s.__name__)

        # checking if the class is actually implementing get_schema and not inheriting it from the
        # parent class or skipping its implementation(forbidded)
        get_schema_implemented = s.get_protobuf_schema.__qualname__.startswith(s.__name__)

        # checking if the class is actually implementing get_wrapped_type and not inheriting it from
        # the parent class or skipping its implementation (forbidden).
        get_wrapped_type_implemented = s.get_wrapped_type.__qualname__.startswith(s.__name__)

        # this is a check to see if the must implement methods were implemented, if not, we cannot
        # serialize this type. Note: we might still be able to serialize its children types, but
        # we handle this in the main function.
        valid_serde = not_abstract and obj2proto_implemented and proto2obj_implemented and get_schema_implemented

        # if its not a valid serde type, return NotSerdeType.
        if not valid_serde:
            return SerdeTypes.NotSerdeType

        # if it is a valid serde type and it implements the get_wrapped method,
        # return SerdeWrapperType.
        if get_wrapped_type_implemented:
            return SerdeTypes.SerdeWrapperType

        # if it is a valid type and it does implement the get_wrapped method, it meants that is
        # a native type, return SerdeNativeType
        return SerdeTypes.SerdeNativeType

    native_types = set()
    wrapper_types = set()

    for s in cls.__subclasses__():
        serde_type = check_type(s)

        if serde_type is SerdeTypes.SerdeNativeType:
            native_types.add(s)

        if serde_type is SerdeTypes.SerdeWrapperType:
            wrapper_types.add(s)

        for c in s.__subclasses__():
            sub_native_set, sub_wrapper_set = get_protobuf(c)
            native_types.union(sub_native_set)
            wrapper_types.union(sub_wrapper_set)

    return native_types, wrapper_types

class Serializable:
    """When we want a custom object to be serializable within the Syft ecosystem
    (as outline in the tutorial above), the first thing we need to do is have it
    subclass from this class. You must then do the following in order for the
    subclass to be properly implemented:

    - implement a protobuf file in the "PySyft/proto" folder for this custom class.
    - compile the protobuf file by running `bash scripts/build_proto`
    - find the generated python file in syft.proto
    - import the generated protobuf class into my custom class
    - set <my class>.protobuf_type = <generated protobuf python class>
    - implement <my class>._object2proto() method to serialize the object to protobuf
    - implement <my class>._proto2object() to deserialize the protobuf object

    At this point, your class should be ready to serialize and deserialize! Don't
    forget to add tests for your object!
    """

    @staticmethod
    def get_protobuf_schema():
        raise NotImplementedError

    @staticmethod
    def get_wrapped_type():
        raise NotImplementedError

    @staticmethod
    def _proto2object(proto):
        raise NotImplementedError

    def _object2proto(self):
        raise NotImplementedError

    def to_proto(self) -> Message:
        return self.serialize(to_proto=True)

    def proto(self) -> Message:
        return self.serialize(to_proto=True)

    def to_json(self) -> str:
        return self.serialize(to_json=True)

    def json(self) -> str:
        return self.serialize(to_json=True)

    def to_binary(self) -> bytes:
        return self.serialize(to_binary=True)

    def binary(self) -> bytes:
        return self.serialize(to_binary=True)

    def to_hex(self) -> str:
        return self.serialize(to_hex=True)

    def hex(self) -> str:
        return self.serialize(to_hex=True)

    def serialize(self, to_proto=True, to_json=False, to_binary=False, to_hex=False):
        """Serialize the object according to the parameters."""

        if to_json or to_binary or to_hex:
            blob = json_format.MessageToJson(message=self._object2proto())

            if to_json:
                return blob

            if to_binary or to_hex:
                blob = bytes(blob, "utf-8")
                if to_hex:
                    blob = blob.hex()
            return blob
        elif to_proto:
            return self._object2proto()
        else:
            raise Exception(
                """You must specify at least one deserialization format using
                            one of the arguments of the serialize() method such as:
                            to_proto, to_json, to_binary, or to_hex."""
            )

def update_serde_cache():
    native_types, wrapper_types = get_protobuf(Serializable)

    for native_type in native_types:
        serde_store.available_types.add(native_type)
        serde_store.qual_name2type[native_type.__qualname__] = native_type
        serde_store.type2schema[native_type] = native_type.get_protobuf_schema()
        serde_store.schema2type[native_type.get_protobuf_schema()] = native_type

    for native_wrapper in wrapper_types:
        serde_store.available_types.add(native_wrapper)
        serde_store.available_types.add(native_wrapper.get_wrapped_type())
        serde_store.wrapped_types.add(native_wrapper.get_wrapped_type())
        serde_store.qual_name2type[native_wrapper.__qualname__] = native_wrapper
        serde_store.type2schema[native_wrapper] = native_wrapper.get_protobuf_schema()
        serde_store.schema2type[native_wrapper.get_protobuf_schema()] = native_wrapper
        serde_store.wrapped2wrapper[native_wrapper.get_wrapped_type()] = native_wrapper
        serde_store.wrapper2wrapper[native_wrapper] = native_wrapper.get_wrapped_type()


@dataclass(frozen=True)
class SerdeStore:
    available_types = LazySet(update_rule=update_serde_cache)
    wrapped_types = LazySet(update_rule=update_serde_cache)
    qual_name2type = LazyDict(update_rule=update_serde_cache)
    type2schema = LazyDict(update_rule=update_serde_cache)
    schema2type = LazyDict(update_rule=update_serde_cache)
    wrapped2wrapper = LazyDict(update_rule=update_serde_cache)
    wrapper2wrapped = LazyDict(update_rule=update_serde_cache)


def _serialize(
    obj: (Serializable, object),
    to_proto=True,
    to_json=False,
    to_binary=False,
    to_hex=False,
):
    if type(obj) not in serde_store.available_types:
        raise TypeError(f"Type {type(obj)} is not serializable by syft! Check how to properly "
                        f"implement serialization in syft/core/common/serializable or in the "
                        f"examples in the docs.")

    if type(obj) in serde_store.wrapped_types:
        wrapper_type = serde_store.wrapped2wrapper[type(obj)]
        wrapper_obj = wrapper_type(obj)
        wrapper_schematic = wrapper_obj.serialize(to_proto=to_proto, to_json=to_json, to_hex=to_hex)
        return wrapper_schematic

    return obj.serialize(
        to_proto=to_proto, to_json=to_json, to_binary=to_binary, to_hex=to_hex
    )


def _deserialize(
    blob: (str, dict, bytes, Message),
    from_proto=True,
    from_json=False,
    from_binary=False,
    from_hex=False,
) -> Serializable:
    """We assume you're deserializing a protobuf object by default"""

    if from_hex:
        from_binary = True
        blob = bytes.fromhex(blob)

    if from_binary:
        from_json = True
        blob = str(blob, "utf-8")

    if from_json:
        from_proto = False
        blob = json.loads(s=blob)
        obj_type_str = blob["objType"]

    if from_proto:
        obj_type_str = blob.obj_type
        proto_obj = blob

    try:
        # lets try to lookup the type we are deserializing
        obj_type = serde_store.schema2type[type(blob)]

    # uh-oh! Looks like the type doesn't exist. Let's throw an informative error.
    except KeyError:

        raise KeyError(
            """You tried to deserialize an unsupported type. This can be caused by
            several reasons. Either you are actively writing Syft code and forgot
            to create one, or you are trying to deserialize an object which was
            serialized using a different version of Syft and the object you tried
            to deserialize is not supported in this version."""
        )

    protobuf_type = serde_store.schema2type[obj_type]

    if not from_proto:
        proto_obj = json_format.ParseDict(js_dict=blob, message=protobuf_type())

    return obj_type._proto2object(proto=proto_obj)

serde_store = SerdeStore()
