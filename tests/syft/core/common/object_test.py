"""In this test suite, we evaluate the ObjectWithID class. For more info
on the ObjectWithID class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways ObjectWithID can/can't be initialized
    - CLASS METHODS: tests for the use of ObjectWithID's class methods
    - SERDE: test for serialization and deserialization of ObjectWithID.
    - CHILDREN: test that subclasses of ObjectWithID fulfill standards

"""

# stdlib
import json
import uuid

# third party
import pytest

# syft absolute
import syft as sy
from syft.core.common import ObjectWithID
from syft.core.common import UID
from syft.util import get_subclasses

# --------------------- INITIALIZATION ---------------------


def test_basic_init() -> None:
    """Test that creating ObjectWithID() does in fact create
    an object with an id."""

    obj = ObjectWithID()
    assert isinstance(obj.id, UID)


def test_immutability_of_id() -> None:
    """We shouldn't allow people to modify the id of an
    ObjectWithID because this can create all sorts of errors.

    Put the other way around - blocking people from modifying
    the ID of an object means we can build a codebase which more
    firmly relies on the id being truthful. It also will avoid
    people initialising objects in weird ways (setting ids later).
    """
    obj = ObjectWithID()

    with pytest.raises(AttributeError):

        # TODO: filter on this error to only include errors
        #  with string "Can't set attribute"

        obj.id = ""


# --------------------- CLASS METHODS ---------------------


def test_compare() -> None:
    """While uses of this feature should be quite rare, we
    should be able to check whether two objects are the same
    based on their IDs being the same by default. Note that
    subclasses will undoubtedly modify this behavior with other
    __eq__ methods."""

    obj = ObjectWithID()
    obj2 = ObjectWithID()

    assert obj != obj2

    obj._id = obj2.id

    assert obj == obj2


def test_to_string() -> None:
    """Tests that UID generates an intuitive string."""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)
    assert str(obj) == "<ObjectWithID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"
    assert obj.__repr__() == "<ObjectWithID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"


# --------------------- SERDE ---------------------


def test_object_with_id_default_serialization() -> None:
    """Tests that default ObjectWithID serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = obj.to_proto()

    assert obj.serialize() == blob


def test_object_with_id_default_deserialization() -> None:
    """Tests that default ObjectWithID deserialization works as expected - from Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = ObjectWithID.get_protobuf_schema()(id=uid.serialize())

    obj2 = sy.deserialize(blob=blob)
    assert obj == obj2


def test_object_with_id_proto_serialization() -> None:
    """Tests that default ObjectWithID serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = ObjectWithID.get_protobuf_schema()(id=uid.serialize())

    assert obj.proto() == blob
    assert obj.to_proto() == blob
    assert obj.serialize(to_proto=True) == blob


def test_object_with_id_proto_deserialization() -> None:
    """Tests that default UID deserialization works as expected - from JSON"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = ObjectWithID.get_protobuf_schema()(id=uid.serialize())

    obj2 = sy.deserialize(blob=blob, from_proto=True)
    assert obj == obj2


def test_object_with_id_json_serialization() -> None:
    """Tests that JSON ObjectWithID serialization works as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}}
    main = {
        "objType": "syft.core.common.object.ObjectWithID",
        "content": json.dumps(content),
    }
    blob = json.dumps(main)

    assert obj.json() == blob
    assert obj.to_json() == blob
    assert obj.serialize(to_json=True) == blob


def test_object_with_id_json_deserialization() -> None:
    """Tests that JSON ObjectWithID deserialization works as expected"""

    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}}
    main = {
        "objType": "syft.core.common.object.ObjectWithID",
        "content": json.dumps(content),
    }
    blob = json.dumps(main)

    obj = sy.deserialize(blob=blob, from_json=True)

    assert obj == ObjectWithID(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    )


def test_object_with_id_binary_serialization() -> None:
    """Tests that binary ObjectWithID serializes as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = (
        b'{"objType": "syft.core.common.object.ObjectWithID", "content":'
        b' "{\\"id\\": {\\"value\\": \\"+xuwZ1u3TEm+zucAqwoVFA==\\"}}"}'
    )

    assert obj.binary() == blob
    assert obj.to_binary() == blob
    assert obj.serialize(to_binary=True) == blob


def test_object_with_id_binary_deserialization() -> None:
    """Test that binary ObjectWithID deserialization works as expected"""

    blob = (
        b'{"objType": "syft.core.common.object.ObjectWithID", "content": '
        b'"{\\"id\\": {\\"value\\": \\"+xuwZ1u3TEm+zucAqwoVFA==\\"}}"}'
    )
    obj = sy.deserialize(blob=blob, from_binary=True)
    assert obj == ObjectWithID(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    )


def test_object_with_id_hex_serialization() -> None:
    """Tests that hex ObjectWithID serializes as expected"""

    obj = ObjectWithID(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    )

    blob = (
        "7b226f626a54797065223a2022737966742e636f72652e636f6d6d6f6e2e6f"
        "626a6563742e4f626a656374576974684944222c2022636f6e74656e74223a20"
        "227b5c2269645c223a207b5c2276616c75655c223a205c222b7875775a3175335"
        "4456d2b7a75634171776f5646413d3d5c227d7d227d"
    )
    assert obj.to_hex() == blob
    assert obj.serialize(to_hex=True) == blob


def test_object_with_id_hex_deserialization() -> None:
    """Test that hex ObjectWithID deserialization works as expected"""

    blob = (
        "7b226f626a54797065223a2022737966742e636f72652e636f6d6d6f6e2e6f"
        "626a6563742e4f626a656374576974684944222c2022636f6e74656e74223a20"
        "227b5c2269645c223a207b5c2276616c75655c223a205c222b7875775a3175335"
        "4456d2b7a75634171776f5646413d3d5c227d7d227d"
    )

    obj = sy.deserialize(blob=blob, from_hex=True)
    assert obj == ObjectWithID(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    )


# ----------------------- CHILDREN -----------------------


def test_subclasses_have_names() -> None:
    """Ensure that all known subclassses of ObjectWithID have
    a __name__ parameter. I'm not sure why but occasionally
    I came across objects without them"""

    subclasses = get_subclasses(obj_type=ObjectWithID)

    for sc in subclasses:
        assert hasattr(sc, "__name__")
