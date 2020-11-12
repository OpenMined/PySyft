# third party
from pytest import raises

# syft absolute
from syft.core.common.serde.deserialize import _deserialize


def test_fail_deserialize_no_format() -> None:
    with raises(TypeError):
        _deserialize(blob="to deserialize", from_proto=False)


def test_fail_deserialize_wrong_format() -> None:
    with raises(TypeError, match="You tried to deserialize an unsupported type."):
        _deserialize(blob="to deserialize")
