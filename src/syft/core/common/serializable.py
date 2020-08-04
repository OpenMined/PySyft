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

# external class/method imports (sorted by length)
from google.protobuf.message import Message
from google.protobuf import json_format

from syft.core.common.lazy_structures import LazyDict

# syft import
from syft.util import get_fully_qualified_name, index_syft_by_module_name


class Serializable(object):
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


    def __init__(self, as_wrapper: bool):
        assert self.protobuf_type is not None
        self.as_wrapper = as_wrapper

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


def _is_string_a_serializable_class_name(lazy_dict, fully_qualified_name: str):

    """This method exists to allow a LazyDict to determine whether an
    object should actually be in its store - aka has the LazyDict been
    lazy and forgotten to add this object thus far.

    In particular, this is the method for the LazyDict within the fully_qualified_name2type
    dictionary - which is used to map fully qualified module names,
    (i.e., 'syft.core.common.UID') to their type object.

    So this method is designed to ask the question, 'Is self_dict an object
    we can serialize?' If it is, we add it to the LazyDict by adding it to
    lazy_dict._dict. If not, we do nothing.

    We determine whether we can serialize the object according to series of
    checks - as outlined below."""

    # lookup the type from the fully qualified name
    # i.e. "syft.core.common.UID" -> <type UID>
    obj_type = index_syft_by_module_name(fully_qualified_name=fully_qualified_name)

    # Check 1: If the object is a subclass of Serializable, then we can serialize it
    # add it to the lazy dictionary.
    if issubclass(obj_type, Serializable):
        lazy_dict._dict[fully_qualified_name] = obj_type

    # Check 2: If the object a non-syft object which is wrapped by a serializable
    # syft object? Aka, since we can't make non-syft objects subclass from
    # Serializable, have we created a wrapper around this object which does
    # subclass from serializable. Note that we can find out by seeing if we
    # monkeypatched a .serializable_wrapper attribute onto this non-syft class.
    elif hasattr(obj_type, "serializable_wrapper_type"):

        # this 'wrapper' object is a syft object which subclasses Serializable
        # so that we can put logic into it showing how to serialize and
        # deserialize the non-syft object.
        wrapper_type = obj_type.serializable_wrapper_type

        # just renaming the variable since we know something about this variable now
        # just so the code reads easier (the compile will remove this so it won't
        # affect performance)
        non_syft_object_fully_qualified_name = fully_qualified_name
        wrapper_type_fully_qualified_name = get_fully_qualified_name(wrapper_type)

        # so now we should update the dictionary so that in the future we can
        # quickly find the wrapper type from both the non_syft_object's fully
        # qualified name and the wrapper_type's fully qualified name
        lazy_dict[wrapper_type_fully_qualified_name] = wrapper_type
        lazy_dict[non_syft_object_fully_qualified_name] = wrapper_type

    else:
        raise Exception(f"{fully_qualified_name} is not serializable")


fully_qualified_name2type = LazyDict(update_rule=_is_string_a_serializable_class_name)


def _serialize(
    obj: (Serializable, object),
    to_proto=True,
    to_json=False,
    to_binary=False,
    to_hex=False,
):

    if not isinstance(obj, Serializable):
        obj = obj.serializable_wrapper_type(value=obj, as_wrapper=True)

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

    global fully_qualified_name2type
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
        obj_type = fully_qualified_name2type[obj_type_str]

    # uh-oh! Looks like the type doesn't exist. Let's throw an informative error.
    except KeyError:

        raise KeyError(
            """You tried to deserialize an unsupported type. This can be caused by
            several reasons. Either you are actively writing Syft code and forgot
            to create one, or you are trying to deserialize an object which was
            serialized using a different version of Syft and the object you tried
            to deserialize is not supported in this version."""
        )

    protobuf_type = obj_type.protobuf_type

    if not from_proto:
        proto_obj = json_format.ParseDict(js_dict=blob, message=protobuf_type())

    return obj_type._proto2object(proto=proto_obj)
