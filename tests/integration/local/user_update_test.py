# stdlib
from typing import TypedDict
from uuid import uuid4

# third party
import pytest

# syft absolute
from syft.orchestra import ClientAlias
from syft.service.response import SyftError
from syft.service.user.user_roles import ServiceRole

# relative
from .conftest import matrix

pytestmark = pytest.mark.local_server


class UserCreateArgs(TypedDict):
    name: str
    email: str
    password: str
    password_verify: str


@pytest.fixture
def user_create_args() -> UserCreateArgs:
    return {
        "name": uuid4().hex,
        "email": f"{uuid4().hex}@example.org",
        "password": (pw := uuid4().hex),
        "password_verify": pw,
    }


@pytest.fixture
def user(client: ClientAlias, user_create_args: UserCreateArgs) -> ClientAlias:
    res = client.register(**user_create_args)
    assert not isinstance(res, SyftError)

    return client.login(
        email=user_create_args["email"],
        password=user_create_args["password"],
    )


@pytest.mark.parametrize("server_args", matrix(port=[None, "auto"]))
def test_user_update_role_str(client: ClientAlias, user: ClientAlias) -> None:
    res = client.users.update(uid=user.account.id, role="admin")
    assert not isinstance(res, SyftError)

    user.refresh()
    assert user.account.role is ServiceRole.ADMIN

    res = user.account.update(role="data_scientist")
    assert not isinstance(res, SyftError)

    user.refresh()
    assert user.account.role is ServiceRole.DATA_SCIENTIST
