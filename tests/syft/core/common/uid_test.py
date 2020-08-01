"""In this test suite, we evaluate the UID class. For more info
on the UID class and its purpose, please see the documentation
in the class itself."""

# external imports
import pytest
import uuid

# syft imports
import syft as sy
from syft.core.common import UID
from syft.core.common.uid import uuid_type


def test_uid_creates_value_if_none_provided():
    """Tests that the UID class will create an ID if none is provided."""

    uid = UID()
    assert uid.value is not None
    assert isinstance(uid.value, uuid_type)


def test_uid_creates_value_if_try_to_init_none():
    """Tests that the UID class will create an ID if you explicitly try to init with None"""

    uid = UID(value=None)
    assert uid.value is not None
    assert isinstance(uid.value, uuid_type)


def test_uid_raises_typeerror_if_string_id_attempted():
    """Tests that the UID class will raise an error if you try to init with a string."""

    with pytest.raises(TypeError) as e:
        uid = UID(value="a string id")


def test_uid_raises_typeerror_if_int_id_attempted():
    """Tests that the UID class will raise an error if you try to init with a string."""

    with pytest.raises(TypeError) as e:
        uid = UID(value=123)


def test_uid_comparison():
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

def test_uid_hash():
    """Tests that a UID hashes correctly. If this tests fails then it
    means that the uuid.UUID library changed or we tried to swap it out
    for something else. Are you sure you want to do this?"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    assert hash(uid) == 1705855162796767136
    assert hash(uid.value.int) == 1705855162796767136

def test_uid_default_serialization():
    """Tests that default UID serialization works as expected - to JSON"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = '{\n  "objType": "syft.core.common.uid.UID",\n  "value": "+xuwZ1u3TEm+zucAqwoVFA=="\n}'
    assert uid.serialize() == blob


def test_uid_binary_serialization():
    """Tests that binary UID serializes as expected"""

    uid = UID(value=uuid.UUID(int=333779996850170035686993356951732753684))
    blob = b'{\n  "objType": "syft.core.common.uid.UID",\n  "value": "+xuwZ1u3TEm+zucAqwoVFA=="\n}'
    assert uid.serialize(to_binary=True) == blob


