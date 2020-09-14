from pytest import raises

from syft.core.common.serde.deserialize import _deserialize


def test_fail_deserialize_no_format():
    with raises(ValueError, match="Please pick the format of the data on the deserialization"):
        _deserialize(blob="to deserialize", from_proto=False)


def test_fail_deserialize_wrong_format():
    with raises(TypeError, match="You tried to deserialize an unsupported type."):
        _deserialize(blob="to deserialize")
