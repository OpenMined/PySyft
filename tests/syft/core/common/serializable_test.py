# third party
import pytest

# syft absolute
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.serialize import _serialize


def test_object_with_no_serialize_wrapper() -> None:
    """
    Test if an object that is Serializable but does not implement the serializable_wrapper_type
    throws an exception when trying to serialize.
    """

    class TestObject(Serializable):
        pass

    with pytest.raises(Exception):
        _serialize(TestObject())
