from typing import List
from typing import Optional
import pytest
import uuid

from syft.exceptions.exception import PySyftException, DEFAULT_PRIVATE_ERROR_MESSAGE
from syft.service.context import AuthedServiceContext
from syft.service.user.user_roles import ServiceRole

TEST_MSG = "PySyftException test message"
TEST_ROLES = [
    ServiceRole.ADMIN,
    ServiceRole.DATA_OWNER,
    ServiceRole.DATA_SCIENTIST,
    ServiceRole.GUEST,
]


def test_PySyftException_raises():
    with pytest.raises(PySyftException) as exc_info:
        raise PySyftException(message=TEST_MSG, roles=[ServiceRole.NONE])

    error_message = exc_info.value.handle()

    assert TEST_MSG in str(error_message)


def test_PySyftException_has_UUID():
    error = PySyftException(message=TEST_MSG)
    error_uuid = error.get_uuid()

    try:
        uuid_obj = uuid.UUID(error_uuid, version=4)
    except ValueError:
        assert False, "Invalid UUID."

    # Check uuid hex
    assert uuid_obj.hex == error.get_uuid().replace("-", "")

    ## Check if reference is dispatched by the handler function
    error_message = error.handle()
    assert error.get_uuid() in str(error_message)


@pytest.mark.parametrize(
    "user_role, roles",
    [
        (ServiceRole.ADMIN, [ServiceRole.GUEST]),
        (ServiceRole.ADMIN, None),
        (ServiceRole.ADMIN, TEST_ROLES),
        (ServiceRole.ADMIN, [ServiceRole.NONE]),
        (ServiceRole.GUEST, [ServiceRole.GUEST]),
        (ServiceRole.DATA_SCIENTIST, TEST_ROLES),
        (ServiceRole.DATA_SCIENTIST, [ServiceRole.GUEST]),
        (ServiceRole.DATA_SCIENTIST, [ServiceRole.DATA_SCIENTIST]),
        (ServiceRole.ADMIN, TEST_ROLES),
        (ServiceRole.ADMIN, TEST_ROLES),
    ],
)
def test_PySyftException_user_reads_error_message_if_allowed(
    authed_context: AuthedServiceContext,
    user_role: ServiceRole,
    roles: Optional[List[ServiceRole]],
):
    authed_context.role = user_role
    error = PySyftException(message=TEST_MSG, roles=roles, context=authed_context)
    error_message = error.handle()

    assert TEST_MSG in str(error_message)


@pytest.mark.parametrize(
    "user_role, roles",
    [
        (ServiceRole.GUEST, None),
        (ServiceRole.DATA_SCIENTIST, [ServiceRole.DATA_OWNER]),
        (ServiceRole.DATA_SCIENTIST, None),
        (ServiceRole.DATA_OWNER, None),
        (ServiceRole.DATA_OWNER, [ServiceRole.ADMIN]),
    ],
)
def test_PySyftException_user_cannot_read_error_message_if_not_allowed(
    authed_context: AuthedServiceContext, user_role, roles: Optional[List[ServiceRole]]
):
    authed_context.role = user_role

    error = PySyftException(message=TEST_MSG, roles=roles, context=authed_context)
    error_message = error.handle()

    assert TEST_MSG not in str(error_message)
    assert DEFAULT_PRIVATE_ERROR_MESSAGE in str(error_message)
