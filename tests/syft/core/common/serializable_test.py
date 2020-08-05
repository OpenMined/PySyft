import uuid
import pytest
from syft.core.common.uid import UID
from syft.core.common.serde.serializable import Serializable
import syft as sy


def test_uuid_wrapper_serialization():
    """A more advanced piece of functionality allows us to
    automatically detect non-syft classes, wrap them and
    serialize them into protobuf. This tests that functoinality
    over uuid objects."""

    uid = uuid.UUID(int=333779996850170035686993356951732753684)

    _uid = UID(value=uid)
    obj_type = UID.__module__ + "." + UID.__name__
    blob = UID.protobuf_type(obj_type=obj_type, value=_uid.value.bytes, as_wrapper=True)

    assert sy.serialize(obj=uid) == blob


def test_uuid_wrapper_deserialization():
    """Test the ability to deserialize correctly into a non-syft type
    using an object which was serialized using a syft-based wrapper."""

    uid = uuid.UUID(int=333779996850170035686993356951732753684)

    _uid = UID(value=uid)
    obj_type = UID.__module__ + "." + UID.__name__
    blob = UID.protobuf_type(obj_type=obj_type, value=_uid.value.bytes, as_wrapper=True)

    assert sy.deserialize(blob=blob) == uid


def test_forgotten_protobuf_type_flag_error():
    """Test whether there is an appropriate warning when someone attempts
    to subclass from Serializable but forgets to put in the protobuf_type
    flag."""

    class CustomSerializable(Serializable):
        def _object2proto(self):
            raise NotImplementedError

        @staticmethod
        def _proto2object(self):
            raise NotImplementedError

    with pytest.raises(TypeError):
        # TODO: tighten this filter to match on the string of the error
        # assert str(e) == "__init__() missing 1 required positional argument: 'as_wrapper'"
        _ = CustomSerializable()

    with pytest.raises(AttributeError):
        # TODO: tighten this filter to match on the string of the error
        # assert str(e) == "'CustomSerializable' object has no attribute 'protobuf_type'"
        _ = CustomSerializable(as_wrapper=False)
