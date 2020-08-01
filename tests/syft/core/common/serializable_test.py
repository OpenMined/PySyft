import syft as sy
import uuid


def test_uuid_wrapper_serialization():
    """A more advanced piece of functionality allows us to
    automatically detect non-syft classes, wrap them and
    serialize them into protobuf. This tests that functoinality
    over uuid objects."""

    uid = uuid.UUID(int=333779996850170035686993356951732753684)
    blob = '{\n  "objType": "syft.core.common.uid.UID",\n  '+\
           '"value": "+xuwZ1u3TEm+zucAqwoVFA==",\n  "asWrapper": true\n}'
    assert sy.serialize(uid) == blob


def test_uuid_wrapper_deserialization():
    """Test the ability to deserialize correctly into a non-syft type
    using an object which was serialized using a syft-based wrapper."""

    uid = uuid.UUID(int=333779996850170035686993356951732753684)
    blob = '{\n  "objType": "syft.core.common.uid.UID",\n  '+\
           '"value": "+xuwZ1u3TEm+zucAqwoVFA==",\n  "asWrapper": true\n}'
    assert sy.deserialize(blob) == uid