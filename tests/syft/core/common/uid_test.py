"""In this test suite, we evaluate the UID class. For more info
on the UID class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways UID can/can't be initialized
    - CLASS METHODS: tests for the use of UID's class methods
    - SERDE: test for serialization and deserialization of UID.

"""

# stdlib
import json
import uuid

# third party
import pytest

# syft absolute
import syft as sy
from syft.core.common import UID
from syft.core.common.serde import _deserialize
from syft.core.common.serde import _serialize
from syft.core.common.uid import uuid_type

# --------------------- INITIALIZATION ---------------------


def test_uid_creates_value_if_none_provided() -> None:
    """Tests that the UID class will create an ID if none is provided."""

    uid = UID()
    assert uid.value is not None
    assert isinstance(uid.value, uuid_type)


def test_uid_creates_value_if_try_to_init_none() -> None:
    """Tests that the UID class will create an ID if you explicitly try to init with None"""

    uid = UID(value=None)
    assert uid.value is not None
    assert isinstance(uid.value, uuid_type)


def test_uid_raises_typeerror_if_string_id_attempted() -> None:
    """Tests that the UID class will raise an error if you try to init with a string."""

    with pytest.raises(TypeError):
        _ = UID(value="a string id")


def test_uid_raises_typeerror_if_int_id_attempted() -> None:
    """Tests that the UID class will raise an error if you try to init with a string."""

    with pytest.raises(TypeError):
        _ = UID(value=123)


# --------------------- CLASS METHODS ---------------------


def test_uid_comparison() -> None:
    """Tests that two UIDs can be compared and will correctly evaluate"""

    uid1 = UID()
    uid2 = UID()

    # first do some basic checks
    assert uid1 == uid1
    assert uid1 != uid2

    # since the comparison should be based on the underlying value
    # let's make sure it's comparing based on the value instead of
    # just based on some other attribute of the object.
    uid2.value = uid1.value
    assert uid1 == uid2


def test_uid_hash() -> None:
    """Tests that a UID hashes correctly. If this tests fails then it
    means that the uuid.UUID library changed or we tried to swap it out
    for something else. Are you sure you want to do this?"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    assert hash(uid) == 1705855162796767136
    assert hash(uid.value.int) == 1705855162796767136

    fake_dict = {}
    fake_dict[uid] = "Just testing we can use it as a key in a dictionary"


def test_to_string() -> None:
    """Tests that UID generates an intuitive string."""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    assert str(uid) == "<UID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"
    assert uid.__repr__() == "<UID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"


def test_from_string() -> None:
    """Tests that UID can be deserialized by a human readable string."""

    uid_str = "fb1bb067-5bb7-4c49-bece-e700ab0a1514"
    uid = UID.from_string(value=uid_str)
    uid_comp = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    assert str(uid) == "<UID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"
    assert uid.__repr__() == "<UID:fb1bb067-5bb7-4c49-bece-e700ab0a1514>"
    assert uid == uid_comp


# --------------------- SERDE ---------------------


def test_uid_default_serialization() -> None:
    """Tests that default UID serialization works as expected - to Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = _serialize(obj=uid)
    _ = _deserialize(blob=blob)
    assert uid.serialize() == blob


def test_uid_default_deserialization() -> None:
    """Tests that default UID deserialization works as expected - from Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = _serialize(obj=uid)
    obj = sy.deserialize(blob=blob)
    assert obj == UID(value=uuid.UUID(int=333779996850170035686993356951732753684))


def test_uid_proto_serialization() -> None:
    """Tests that proto UID serialization works as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    blob = _serialize(obj=uid)

    assert uid.proto() == blob
    assert uid.to_proto() == blob
    assert uid.serialize(to_proto=True) == blob


def test_uid_proto_deserialization() -> None:
    """Tests that proto UID deserialization works as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = _serialize(obj=uid)

    obj = sy.deserialize(blob=blob, from_proto=True)
    assert obj == UID(value=uuid.UUID(int=333779996850170035686993356951732753684))


def test_uid_json_serialization() -> None:
    """Tests that JSON UID serialization works as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    content = {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}
    main = {"objType": "syft.core.common.uid.UID", "content": json.dumps(content)}
    blob = json.dumps(main)

    assert uid.json() == blob
    assert uid.to_json() == blob
    assert uid.serialize(to_json=True) == blob


def test_uid_json_deserialization() -> None:
    """Tests that JSON UID deserialization works as expected"""

    content = {"value": "+xuwZ1u3TEm+zucAqwoVFA=="}
    main = {"objType": "syft.core.common.uid.UID", "content": json.dumps(content)}

    blob = json.dumps(main)
    obj = sy.deserialize(blob=blob, from_json=True)
    assert obj == UID(value=uuid.UUID(int=333779996850170035686993356951732753684))


def test_uid_binary_serialization() -> None:
    """Tests that binary UID serializes as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = (
        b'{"objType": "syft.core.common.uid.UID", "content": '
        b'"{\\"value\\": \\"+xuwZ1u3TEm+zucAqwoVFA==\\"}"}'
    )

    assert uid.binary() == blob
    assert uid.to_binary() == blob
    assert uid.serialize(to_binary=True) == blob


def test_uid_binary_deserialization() -> None:
    """Test that binary deserialization works as expected"""

    blob = (
        b'{"objType": "syft.core.common.uid.UID", '
        b'"content": "{\\"value\\": \\"+xuwZ1u3TEm+zucAqwoVFA==\\"}"}'
    )

    obj = sy.deserialize(blob=blob, from_binary=True)
    assert obj == UID(value=uuid.UUID(int=333779996850170035686993356951732753684))


def test_uid_hex_serialization() -> None:
    """Tests that hex UID serializes as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = (
        "7b226f626a54797065223a2022737966742e636f72652e636f6d6d6f6e2e7569"
        "642e554944222c2022636f6e74656e74223a20227b5c2276616c75655c223a205c2"
        "22b7875775a31753354456d2b7a75634171776f5646413d3d5c227d227d"
    )
    assert uid.to_hex() == blob
    assert uid.serialize(to_hex=True) == blob


def test_uid_hex_deserialization() -> None:
    """Test that hex deserialization works as expected"""

    blob = (
        "7b226f626a54797065223a2022737966742e636f72652e636f6d6d6f6e2e7569"
        "642e554944222c2022636f6e74656e74223a20227b5c2276616c75655c223a205c2"
        "22b7875775a31753354456d2b7a75634171776f5646413d3d5c227d227d"
    )
    obj = sy.deserialize(blob=blob, from_hex=True)
    assert obj == UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
