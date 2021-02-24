"""In this test suite, we evaluate the ObjectWithID class. For more info
on the ObjectWithID class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways ObjectWithID can/can't be initialized
    - CLASS METHODS: tests for the use of ObjectWithID's class methods
    - SERDE: test for serialization and deserialization of ObjectWithID.
    - CHILDREN: test that subclasses of ObjectWithID fulfil standards

"""

# stdlib
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
        #  with the string "Can't set attribute"

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
    assert str(obj) == "<ObjectWithID: fb1bb0675bb74c49becee700ab0a1514>"
    assert obj.__repr__() == "<ObjectWithID: fb1bb0675bb74c49becee700ab0a1514>"


# --------------------- SERDE ---------------------


def test_object_with_id_default_serialization() -> None:
    """Tests that default ObjectWithID serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = sy.serialize(obj, to_proto=True)

    assert sy.serialize(obj) == blob


def test_object_with_id_default_deserialization() -> None:
    """Tests that default ObjectWithID deserialization works as expected - from Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = ObjectWithID.get_protobuf_schema()(id=sy.serialize(uid))

    obj2 = sy.deserialize(blob=blob)
    assert obj == obj2


def test_object_with_id_proto_serialization() -> None:
    """Tests that default ObjectWithID serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = ObjectWithID.get_protobuf_schema()(id=sy.serialize(uid))

    assert sy.serialize(obj, to_proto=True) == blob
    assert sy.serialize(obj, to_proto=True) == blob
    assert sy.serialize(obj, to_proto=True) == blob


def test_object_with_id_proto_deserialization() -> None:
    """Tests that default UID deserialization works as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = ObjectWithID.get_protobuf_schema()(id=sy.serialize(uid))

    obj2 = sy.deserialize(blob=blob, from_proto=True)
    assert obj == obj2


def test_object_with_id_binary_serialization() -> None:
    """Tests that binary ObjectWithID serializes as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = ObjectWithID(id=uid)

    blob = (
        b"\n$syft.core.common.object.ObjectWithID\x12\x14\n\x12\n\x10\xfb\x1b\xb0"
        + b"g[\xb7LI\xbe\xce\xe7\x00\xab\n\x15\x14"
    )

    assert sy.serialize(obj, to_bytes=True) == blob
    assert sy.serialize(obj, to_bytes=True) == blob
    assert sy.serialize(obj, to_bytes=True) == blob


def test_object_with_id_binary_deserialization() -> None:
    """Test that binary ObjectWithID deserialization works as expected"""

    blob = (
        b"\n$syft.core.common.object.ObjectWithID\x12\x14\n\x12\n\x10\xfb\x1b\xb0"
        + b"g[\xb7LI\xbe\xce\xe7\x00\xab\n\x15\x14"
    )
    obj = sy.deserialize(blob=blob, from_bytes=True)
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
