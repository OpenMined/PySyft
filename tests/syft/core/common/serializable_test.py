import uuid
from syft.core.common.uid import UID

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

    assert sy.serialize(uid) == blob


def test_uuid_wrapper_deserialization():
    """Test the ability to deserialize correctly into a non-syft type
    using an object which was serialized using a syft-based wrapper."""

    uid = uuid.UUID(int=333779996850170035686993356951732753684)

    _uid = UID(value=uid)
    obj_type = UID.__module__ + "." + UID.__name__
    blob = UID.protobuf_type(obj_type=obj_type, value=_uid.value.bytes, as_wrapper=True)

    assert sy.deserialize(blob) == uid
