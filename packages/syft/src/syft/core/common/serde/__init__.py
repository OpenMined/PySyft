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
    certainly, be slower than using technology like protobuf,
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
    # implies it's literally just an object with an ID.
    # We'll use it to show serialization, but you can use the
    # same approach with any serializable object in the Syft
    # ecosystem.
    from syft.core.common.object import ObjectWithID

    # this creates an object which has an id
    my_object = ObjectWithID()

    print(my_object)
    # >>> <ObjectWithID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>

    # by default, serialize() will serialize it to a protobuf Message object
    proto_obj = serialize(my_object)

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
convenience functions for 4 popular representations: protobuf, json, binary,
and hex. Setup::

    import syft as sy
    from syft.core.common.object import ObjectWithID

    obj = ObjectWithId()

Protobuf
********

.. code::

    proto_obj = serialize(obj, to_proto=True)
    proto_obj = serialize(obj, to_proto=True)
    proto_obj = serialize(obj, to_proto=True)

    print(proto_obj)
    # >>> obj_type: "syft.core.common.object.ObjectWithID"
    # >>> id {
    # >>>   obj_type: "syft.core.common.uid.UID"
    # >>>   value: "23hi23hgo2ih23ih2;o3igh;2oih;iagapwihpag"
    # >>> }

    obj_again = sy.deserialize(blob=proto_obj, from_proto=True)

Binary
******

.. code::

    binary_obj = serialize(obj, to_bytes=True)
    binary_obj = serialize(obj, to_bytes=True)
    binary_obj = serialize(obj, to_bytes=True)

    # print(binary_obj)
    # >>> b'{  "objType": "syft.core.common.object.ObjectWithID",
    # >>> "id": {    "objType": "syft.core.common.uid.UID",
    # >>> "value": "+xuwZ1u3TEm+zucAqwoVFA=="  }}'

    obj_again = sy.deserialize(blob=proto_obj, from_bytes=True)

Now we can continue with the class definition for the Serializable class, which
is the parent class for all serializable objects within Syft.

If you'd like to see a simple example of a class which can be serialized, please read
the source code of :py:mod:`syft.core.common.object.ObjectWithID`.
"""

# relative
from .deserialize import _deserialize  # noqa: F401
from .serialize import _serialize  # noqa: F401
