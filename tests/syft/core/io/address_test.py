"""In this test suite, we evaluate the Address class. For more info
on the Address class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways Address can/can't be initialized
    - CLASS METHODS: tests for the use of Address's class methods
    - SERDE: test for serialization and deserialization of Address.

"""


# external imports
import uuid
import json
import pytest

# syft imports
from syft.core.io.location.specific import SpecificLocation
from syft.core.common.uid import UID
from syft.core.io.address import Address
import syft as sy

# --------------------- INITIALIZATION ---------------------


def test_init_without_arguments():
    """Test that Address have all attributes as None if none are given"""

    # init works without arguments
    addr = Address()

    assert addr.network is None
    assert addr.domain is None
    assert addr.device is None
    assert addr.vm is None

    with pytest.raises(Exception):
        assert addr.target_id is None


def test_init_with_specific_id():
    """Test that Address will use the SpecificLocation you pass into the constructor"""

    # init works with arguments
    addr = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    assert addr.network is not None
    assert addr.domain is not None
    assert addr.device is not None
    assert addr.vm is not None

    # init works without arguments
    addr = Address(  # network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    assert addr.network is None
    assert addr.domain is not None
    assert addr.device is not None
    assert addr.vm is not None

    # init works without arguments
    addr = Address(
        network=SpecificLocation(id=UID()),
        # domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    assert addr.network is not None
    assert addr.domain is None
    assert addr.device is not None
    assert addr.vm is not None

    # init works without arguments
    addr = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        # device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    assert addr.network is not None
    assert addr.domain is not None
    assert addr.device is None
    assert addr.vm is not None

    # init works without arguments
    addr = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        # vm=SpecificLocation(id=UID())
    )

    assert addr.network is not None
    assert addr.domain is not None
    assert addr.device is not None
    assert addr.vm is None


# --------------------- CLASS METHODS ---------------------


def test_compare():
    """Tests whether two address objects are the same. This functionality
    is likely to get used a lot when nodes are determining whether a message
    is for them or not."""

    x = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    y = Address(
        network=SpecificLocation(id=UID()),
        domain=SpecificLocation(id=UID()),
        device=SpecificLocation(id=UID()),
        vm=SpecificLocation(id=UID()),
    )

    z = Address(network=x.network, domain=x.domain, device=x.device, vm=x.vm)

    assert x != y
    assert x == z
    assert y != z


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
    obj = SpecificLocation(id=uid)

    blob = obj.to_proto()

    assert obj.serialize() == blob


def test_default_deserialization():
    """Tests that default SpecificLocation deserialization works as expected - from Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid)

    blob = SpecificLocation.get_protobuf_schema()(id=uid.serialize())

    obj2 = sy.deserialize(blob=blob)
    assert obj == obj2


def test_proto_serialization():
    """Tests that default SpecificLocation serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    obj = SpecificLocation(id=uid)

    blob = SpecificLocation.get_protobuf_schema()(id=uid.serialize())

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
    obj = SpecificLocation(id=uid)

    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}}
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

    content = {"id": {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}}
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
    obj = SpecificLocation(id=uid)

    blob = (
        b'{"objType": "syft.core.io.location.specific.SpecificLocation", "content":'
        b' "{\\"id\\": {\\"value\\": \\"+xuwZ1u3TEm+zucAqwoVFA==\\"}}"}'
    )

    assert obj.binary() == blob
    assert obj.to_binary() == blob
    assert obj.serialize(to_binary=True) == blob


def test_binary_deserialization():
    """Test that binary SpecificLocation deserialization works as expected"""

    blob = (
        b'{"objType": "syft.core.io.location.specific.SpecificLocation", "content": '
        b'"{\\"id\\": {\\"value\\": \\"+xuwZ1u3TEm+zucAqwoVFA==\\"}}"}'
    )
    obj = sy.deserialize(blob=blob, from_binary=True)
    assert obj == SpecificLocation(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    )


def test_hex_serialization():
    """Tests that hex SpecificLocation serializes as expected"""

    obj = SpecificLocation(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    )

    blob = (
        "7b226f626a54797065223a2022737966742e636f72652e696f2e6c6f636174696"
        "f6e2e73706563696669632e53706563696669634c6f636174696f6e222c202263"
        "6f6e74656e74223a20227b5c2269645c223a207b5c2276616c75655c223a205c2"
        "22b7875775a31753354456d2b7a75634171776f5646413d3d5c227d7d227d"
    )

    assert obj.hex() == blob
    assert obj.to_hex() == blob
    assert obj.serialize(to_hex=True) == blob


def test_hex_deserialization():
    """Test that hex SpecificLocation deserialization works as expected"""

    blob = (
        "7b226f626a54797065223a2022737966742e636f72652e696f2e6c6f636174696"
        "f6e2e73706563696669632e53706563696669634c6f636174696f6e222c202263"
        "6f6e74656e74223a20227b5c2269645c223a207b5c2276616c75655c223a205c2"
        "22b7875775a31753354456d2b7a75634171776f5646413d3d5c227d7d227d"
    )

    obj = sy.deserialize(blob=blob, from_hex=True)
    assert obj == SpecificLocation(
        id=UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    )
