# stdlib
from unittest.mock import Mock

# third party
import pytest

# syft absolute
from syft.service.context import AuthedServiceContext
from syft.service.user.user_roles import ServiceRole
from syft.types.errors import SyftException

default_public_message = SyftException.public_message
public_message = "An error occurred. Contact the admin for more information."
private_message = "Private admin error."


def test_default_public_message():
    default_public_message = SyftException.public_message
    exception = SyftException(private_message)

    assert exception.public == default_public_message
    assert exception._private_message == private_message


def test_custom_public_message():
    exception = SyftException(private_message, public_message=public_message)

    assert exception.public == public_message
    assert exception._private_message == private_message


def test_public_message_property():
    default_public_message = SyftException.public_message
    exception = SyftException(private_message)

    assert exception.public == default_public_message


@pytest.mark.parametrize(
    "role,private_msg,public_msg,expected_message",
    [
        (ServiceRole.NONE, private_message, None, default_public_message),
        (ServiceRole.GUEST, private_message, None, default_public_message),
        (ServiceRole.DATA_SCIENTIST, private_message, None, default_public_message),
        (ServiceRole.DATA_OWNER, private_message, None, private_message),
        (ServiceRole.ADMIN, private_message, None, private_message),
    ],
)
def test_get_message(role, private_msg, public_msg, expected_message):
    mock_context = Mock(AuthedServiceContext)
    mock_context.role = role
    mock_context.dev_mode = False
    exception = SyftException(private_msg, public_message=public_msg)
    assert exception.get_message(mock_context) == expected_message
