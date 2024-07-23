# stdlib
from typing import Any

# third party
import pytest
from pytest import FixtureRequest

# syft absolute
import syft as sy


@pytest.mark.parametrize(
    "obj",
    [
        "settings",
        "update_settings",
        "metadata_json",
    ],
)
def test_server_settings_serde(obj: Any, request: FixtureRequest) -> None:
    requested_obj = request.getfixturevalue(obj)
    ser_data = sy.serialize(requested_obj, to_bytes=True)
    assert isinstance(ser_data, bytes)

    deser_data = sy.deserialize(ser_data, from_bytes=True)
    assert isinstance(deser_data, type(requested_obj))
    assert deser_data == requested_obj
