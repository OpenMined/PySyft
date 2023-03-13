# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.parametrize(
    "obj",
    [
        "guest_create_user",
        "guest_view_user",
        "guest_user",
        "guest_user_private_key",
        "update_user",
        "guest_user_search",
        "user_stash",
        "user_service",
    ],
)
def test_user_serde(obj):
    ser_data = sy.serialize(obj, to_bytes=True)
    assert isinstance(ser_data, bytes)
    deser_data = sy.deserialize(ser_data, from_bytes=True)

    assert isinstance(deser_data, type(obj))
    assert deser_data == obj
