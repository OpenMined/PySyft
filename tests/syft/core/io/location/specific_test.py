"""In this test suite, we evaluate the SpecificLocation class. For more info
on the SpecificLocation class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways SpecificLocation can/can't be initialized
    - CLASS METHODS: tests for the use of SpecificLocation's class methods
    - SERDE: test for serialization and deserialization of SpecificLocation.

"""


# external imports
import uuid
import json

# syft imports
from syft.core.io.location.specific import SpecificLocation
from syft.core.common.uid import UID
import syft as sy

# --------------------- INITIALIZATION ---------------------


def test_specific_location_init_without_arguments():
    """Test that SpecificLocation will self-create an ID object if none is given"""

    # init works without arguments
    loc = SpecificLocation()

    assert isinstance(loc.id, UID)


def test_specific_location_init_with_specific_id():
    """Test that SpecificLocation will use the ID you pass into the constructor"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    loc = SpecificLocation(id=uid)

    assert loc.id == uid


# --------------------- CLASS METHODS ---------------------


def test_compare():
    """While uses of this feature should be quite rare, we
    should be able to check whether two objects are the same
    based on their IDs being the same by default. Note that
    subclasses will undoubtedly modify this behavior with other
    __eq__ methods."""

    obj = SpecificLocation()
    obj2 = SpecificLocation()

    assert obj != obj2

    obj._id = obj2.id

    assert obj == obj2


def test_to_string():
    """Tests that SpecificLocation generates an intuitive string."""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid)
    assert str(obj) == "<SpecificLocation:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"
    assert obj.__repr__() == "<SpecificLocation:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"


# --------------------- SERDE ---------------------


def test_default_serialization():
    """Tests that default SpecificLocation serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid, name="Test")

    blob = obj.to_proto()

    assert obj.serialize() == blob


def test_default_deserialization():
    """Tests that default SpecificLocation deserialization works as expected - from Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid, name="Test")

    blob = SpecificLocation.get_protobuf_schema()(id=uid.serialize())

    obj2 = sy.deserialize(blob=blob)
    assert obj == obj2


def test_proto_serialization():
    """Tests that default SpecificLocation serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid, name="Test")

    blob = SpecificLocation.get_protobuf_schema()(id=uid.serialize(), name="Test")

    assert obj.proto() == blob
    assert obj.to_proto() == blob
    assert obj.serialize(to_proto=True) == blob


def test_proto_deserialization():
    """Tests that default SpecificLocation deserialization works as expected - from Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid)

    blob = SpecificLocation.get_protobuf_schema()(id=uid.serialize())

    obj2 = sy.deserialize(blob=blob, from_proto=True)
    assert obj == obj2


def test_json_serialization():
    """Tests that JSON SpecificLocation serialization works as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid, name="Test")

    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}, "name": "Test"}
    main = {
        "objType": "syft.core.io.location.specific.SpecificLocation",
        "content": json.dumps(content),
    }
    blob = json.dumps(main)

    assert obj.json() == blob
    assert obj.to_json() == blob
    assert obj.serialize(to_json=True) == blob


def test_json_deserialization():
    """Tests that JSON SpecificLocation deserialization works as expected"""

    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}, "name": "Test"}
    main = {
        "objType": "syft.core.io.location.specific.SpecificLocation",
        "content": json.dumps(content),
    }
    blob = json.dumps(main)

    obj = sy.deserialize(blob=blob, from_json=True)

    assert obj == SpecificLocation(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    )


def test_binary_serialization():
    """Tests that binary SpecificLocation serializes as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid, name="Test")

    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}, "name": "Test"}
    main = {
        "objType": "syft.core.io.location.specific.SpecificLocation",
        "content": json.dumps(content),
    }
    blob = bytes(json.dumps(main), "utf-8")

    assert obj.binary() == blob
    assert obj.to_binary() == blob
    assert obj.serialize(to_binary=True) == blob


def test_binary_deserialization():
    """Test that binary SpecificLocation deserialization works as expected"""

    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}, "name": "Test"}
    main = {
        "objType": "syft.core.io.location.specific.SpecificLocation",
        "content": json.dumps(content),
    }
    blob = bytes(json.dumps(main), "utf-8")

    obj = sy.deserialize(blob=blob, from_binary=True)
    assert obj == SpecificLocation(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684)),
        name="Test",
    )


def test_hex_serialization():
    """Tests that hex SpecificLocation serializes as expected"""

    obj = SpecificLocation(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684)),
        name="Test",
    )

    blob = (
        "7b226f626a54797065223a2022737966742e636f72652e696f2e6c6f636174696f6e2e73706563"
        "696669632e53706563696669634c6f636174696f6e222c2022636f6e74656e74223a20227b5c22"
        "69645c223a207b5c2276616c75655c223a205c222b7875775a31753354456d2b7a75634171776f"
        "5646413d3d5c227d2c205c226e616d655c223a205c22546573745c227d227d"
    )

    assert obj.hex() == blob
    assert obj.to_hex() == blob
    assert obj.serialize(to_hex=True) == blob


def test_hex_deserialization():
    """Test that hex SpecificLocation deserialization works as expected"""

    blob = (
        "7b226f626a54797065223a2022737966742e636f72652e696f2e6c6f636174696f6e2e73706563"
        "696669632e53706563696669634c6f636174696f6e222c2022636f6e74656e74223a20227b5c22"
        "69645c223a207b5c2276616c75655c223a205c222b7875775a31753354456d2b7a75634171776f"
        "5646413d3d5c227d2c205c226e616d655c223a205c22546573745c227d227d"
    )

    obj = sy.deserialize(blob=blob, from_hex=True)
    assert obj == SpecificLocation(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684)),
        name="Test",
    )
