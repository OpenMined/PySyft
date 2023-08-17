"""In this test suite, we evaluate the UID class. For more info
on the UID class and its purpose, please see the documentation
in the class itself.

Table of Contents:
    - INITIALIZATION: tests for various ways UID can/can't be initialized
    - CLASS METHODS: tests for the use of UID's class methods
    - SERDE: test for serialization and deserialization of UID.

"""

# stdlib
import uuid

# third party
import pytest

# syft absolute
import syft as sy
from syft.serde.serialize import _serialize
from syft.types.uid import UID
from syft.types.uid import uuid_type


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
    """Tests that a UID hashes correctly. If this test fails then it
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

    assert str(uid) == "fb1bb0675bb74c49becee700ab0a1514"
    assert uid.__repr__() == "<UID: fb1bb0675bb74c49becee700ab0a1514>"


def test_from_string() -> None:
    """Tests that UID can be deserialized by a human readable string."""

    uid_str = "fb1bb067-5bb7-4c49-bece-e700ab0a1514"
    uid = UID.from_string(value=uid_str)
    uid_comp = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    assert str(uid) == "fb1bb0675bb74c49becee700ab0a1514"
    assert uid.__repr__() == "<UID: fb1bb0675bb74c49becee700ab0a1514>"
    assert uid == uid_comp


def test_from_string_exception() -> None:
    """Tests that UID throws exception when invalid string is given."""

    with pytest.raises(ValueError):
        UID.from_string(value="Hello world")


# --------------------- SERDE ---------------------
def test_uid_default_deserialization() -> None:
    """Tests that default UID deserialization works as expected - from Protobuf"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = _serialize(obj=uid)
    obj = sy.deserialize(blob=blob)
    assert obj == UID(value=uuid.UUID(int=333779996850170035686993356951732753684))


def test_uid_proto_serialization() -> None:
    """Tests that proto UID serialization works as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))

    blob = _serialize(obj=uid, to_bytes=True)

    assert sy.serialize(uid, to_bytes=True) == blob
    assert sy.serialize(uid, to_bytes=True) == blob
    assert sy.serialize(uid, to_bytes=True) == blob


def test_uid_proto_deserialization() -> None:
    """Tests that proto UID deserialization works as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = _serialize(obj=uid)

    obj = sy.deserialize(blob=blob, from_proto=True)
    assert obj == UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
