# stdlib
from typing import Literal
from typing import NoReturn
from unittest import mock

# third party
from faker import Faker
import pytest
from pytest import MonkeyPatch

# syft absolute
import syft as sy
from syft import orchestra
from syft.client.client import SyftClient
from syft.server.credentials import SyftVerifyKey
from syft.server.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.context import ServerServiceContext
from syft.service.context import UnauthedServiceContext
from syft.service.response import SyftSuccess
from syft.service.user import errors as user_errors
from syft.service.user.user import User
from syft.service.user.user import UserCreate
from syft.service.user.user import UserPrivateKey
from syft.service.user.user import UserUpdate
from syft.service.user.user import UserView
from syft.service.user.user_roles import ServiceRole
from syft.service.user.user_service import UserService
from syft.store.document_store_errors import NotFoundException
from syft.store.document_store_errors import StashException
from syft.types.errors import SyftException
from syft.types.result import Ok
from syft.types.result import as_result
from syft.types.uid import UID


def settings_with_signup_enabled(worker) -> type:
    mock_settings = worker.settings
    mock_settings.signup_enabled = True

    return mock_settings


def test_userservice_create_when_user_exists(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_create_user: UserCreate,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> User:
        return guest_create_user.to(User)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    with pytest.raises(SyftException):
        user_service.create(authed_context, **guest_create_user)


def test_userservice_create_error_on_get_by_email(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_create_user: UserCreate,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> User:
        return guest_create_user.to(User)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    with pytest.raises(SyftException) as exc:
        user_service.create(authed_context, **guest_create_user)

    assert exc.value.public_message == f"User {guest_create_user.email} already exists"


def test_userservice_create_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_create_user: UserCreate,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> User:
        raise NotFoundException

    expected_user = guest_create_user.to(User)
    expected_output: UserView = expected_user.to(UserView)
    expected_output.syft_client_verify_key = authed_context.credentials
    expected_output.syft_server_location = authed_context.server.id

    @as_result(StashException)
    def mock_set(
        credentials: SyftVerifyKey,
        obj: User,
        has_permission: bool = False,
        add_permissions=None,
    ) -> User:
        return expected_user

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    monkeypatch.setattr(user_service.stash, "set", mock_set)

    response = user_service.create(authed_context, **guest_create_user)
    assert isinstance(response, UserView)
    assert response.model_dump() == expected_output.model_dump()


def test_userservice_create_error_on_set(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_create_user: UserCreate,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> NoReturn:
        raise NotFoundException

    @as_result(StashException)
    def mock_set(
        credentials: SyftVerifyKey,
        obj: User,
        has_permission: bool = False,
        add_permissions=None,
    ) -> NoReturn:
        raise StashException

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    monkeypatch.setattr(user_service.stash, "set", mock_set)

    with pytest.raises(StashException) as exc:
        user_service.create(authed_context, **guest_create_user)

    assert exc.type == StashException


def test_userservice_view_error_on_get_by_uid(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
) -> None:
    uid_to_view = UID()
    expected_error_msg = f"Item {uid_to_view} not found"

    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> NoReturn:
        raise NotFoundException(public_message=expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    with pytest.raises(NotFoundException) as exc:
        user_service.view(authed_context, uid_to_view)
    assert exc.type == NotFoundException
    assert exc.value.public_message == expected_error_msg


def test_userservice_view_user_not_exists(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
) -> None:
    uid_to_view = UID()

    expected_error_msg = f"User {uid_to_view} not found"

    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> NoReturn:
        raise NotFoundException(public_message=expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    with pytest.raises(NotFoundException) as exc:
        user_service.view(authed_context, uid_to_view)

    assert exc.type == NotFoundException
    assert exc.value.public == expected_error_msg


def test_userservice_view_user_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
) -> None:
    uid_to_view = guest_user.id

    expected_output = guest_user.to(UserView)

    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> User:
        return guest_user

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    response = user_service.view(authed_context, uid=uid_to_view)

    assert isinstance(response, UserView)
    assert response.model_dump() == expected_output.model_dump()


def test_userservice_get_all_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
    admin_user: User,
) -> None:
    mock_get_all_output = [guest_user, admin_user]
    expected_output = [x.to(UserView) for x in mock_get_all_output]

    @as_result(StashException)
    def mock_get_all(credentials: SyftVerifyKey, **kwargs) -> list[User]:
        return mock_get_all_output

    monkeypatch.setattr(user_service.stash, "get_all", mock_get_all)

    response = user_service.get_all(authed_context)
    assert isinstance(response, list)
    assert len(response) == len(expected_output)
    assert all(
        r.model_dump() == expected.model_dump()
        for r, expected in zip(response, expected_output)
    )


def test_userservice_search(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
) -> None:
    @as_result(SyftException)
    def get_all(credentials: SyftVerifyKey, **kwargs) -> list[User]:
        for key in kwargs.keys():
            if hasattr(guest_user, key):
                return [guest_user]
        return []

    monkeypatch.setattr(user_service.stash, "get_all", get_all)

    expected_output = [guest_user.to(UserView)]

    # Search via id
    response = user_service.search(context=authed_context, id=guest_user.id)

    assert isinstance(response, list)
    assert all(
        r.to_dict() == expected.to_dict()
        for r, expected in zip(response, expected_output)
    )
    # assert response.to_dict() == expected_output.to_dict()

    # Search via email
    response = user_service.search(context=authed_context, email=guest_user.email)
    assert isinstance(response, list)
    assert all(
        r.to_dict() == expected.to_dict()
        for r, expected in zip(response, expected_output)
    )

    # Search via name
    response = user_service.search(context=authed_context, name=guest_user.name)
    assert isinstance(response, list)
    assert all(
        r.to_dict() == expected.to_dict()
        for r, expected in zip(response, expected_output)
    )

    # Search via verify_key
    response = user_service.search(
        context=authed_context,
        verify_key=guest_user.verify_key,
    )
    assert isinstance(response, list)
    assert all(
        r.to_dict() == expected.to_dict()
        for r, expected in zip(response, expected_output)
    )

    # Search via multiple kwargs
    response = user_service.search(
        context=authed_context, name=guest_user.name, email=guest_user.email
    )
    assert isinstance(response, list)
    assert all(
        r.to_dict() == expected.to_dict()
        for r, expected in zip(response, expected_output)
    )


def test_userservice_search_with_invalid_kwargs(
    worker, user_service: UserService, authed_context: AuthedServiceContext
) -> None:
    # Direct calls will fail with a type error
    with pytest.raises(TypeError) as exc:
        user_service.search(context=authed_context, role=ServiceRole.GUEST)

    assert "UserService.search() got an unexpected keyword argument 'role'" == str(
        exc.value
    )

    root_client = worker.root_client
    # Client calls fails at autosplat check
    with pytest.raises(SyftException) as exc:
        root_client.users.search(role=ServiceRole.GUEST)

    assert "Invalid parameter: `role`" in exc.value.public_message


def test_userservice_update_get_by_uid_fails(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    update_user: UserUpdate,
) -> None:
    random_uid = UID()
    expected_error_msg = f"User {random_uid} not found"

    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> NoReturn:
        raise NotFoundException(public_message=expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    with pytest.raises(NotFoundException) as exc:
        user_service.update(authed_context, uid=random_uid, **update_user)

    assert exc.type == NotFoundException
    assert exc.value.public == expected_error_msg


def test_userservice_update_no_user_exists(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    update_user: UserUpdate,
) -> None:
    random_uid = UID()
    expected_error_msg = f"User {random_uid} not found"

    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> NoReturn:
        raise NotFoundException(public_message=expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    with pytest.raises(NotFoundException) as exc:
        user_service.update(authed_context, uid=random_uid, **update_user)

    assert exc.type == NotFoundException
    assert exc.value.public_message == expected_error_msg


def test_userservice_update_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
    update_user: UserUpdate,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> User:
        return guest_user

    @as_result(NotFoundException)
    def mock_update(
        credentials: SyftVerifyKey, obj: User, has_permission: bool
    ) -> User:
        guest_user.name = obj.name
        guest_user.email = obj.email
        return guest_user

    monkeypatch.setattr(user_service.stash, "update", mock_update)
    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    authed_context.role = ServiceRole.ADMIN
    user = user_service.update(authed_context, uid=guest_user.id, **update_user)

    assert isinstance(user, UserView)
    assert user.email == update_user.email
    assert user.name == update_user.name

    another_update = UserUpdate(name="name", email="email@openmined.org")
    user = user_service.update(authed_context, guest_user.id, **another_update)
    assert isinstance(user, UserView)
    assert user.name == "name"
    assert user.email == "email@openmined.org"


def test_userservice_update_fails(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
    update_user: UserUpdate,
) -> None:
    update_error_msg = "Failed to reach server."

    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> User:
        return guest_user

    @as_result(StashException)
    def mock_update(
        credentials: SyftVerifyKey, obj: User, has_permission: bool
    ) -> NoReturn:
        raise StashException(update_error_msg)

    monkeypatch.setattr(user_service.stash, "update", mock_update)
    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    authed_context.role = ServiceRole.ADMIN

    with pytest.raises(StashException) as exc:
        user_service.update(authed_context, uid=guest_user.id, **update_user)

    assert exc.type == StashException
    assert exc.value.public == StashException.public_message
    assert exc.value._private_message == update_error_msg
    assert exc.value.get_message(authed_context) == update_error_msg


def test_userservice_delete_failure(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
) -> None:
    id_to_delete = UID()

    expected_error_msg = f"User {id_to_delete} not found"

    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> NoReturn:
        raise NotFoundException(public_message=expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    with pytest.raises(NotFoundException) as exc:
        user_service.delete(context=authed_context, uid=id_to_delete)

    assert exc.type == NotFoundException
    assert exc.value.public == expected_error_msg

    @as_result(NotFoundException)
    def mock_get_by_uid_good(credentials: SyftVerifyKey, uid: UID) -> User:
        return guest_user

    @as_result(user_errors.UserDeleteError)
    def mock_delete_by_uid(
        credentials: SyftVerifyKey, uid: UID, has_permission=False
    ) -> NoReturn:
        raise user_errors.UserDeleteError(public_message=expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid_good)
    monkeypatch.setattr(user_service.stash, "delete_by_uid", mock_delete_by_uid)

    with pytest.raises(user_errors.UserPermissionError) as exc:
        user_service.delete(context=authed_context, uid=id_to_delete)

    assert exc.type == user_errors.UserPermissionError
    assert exc.value._private_message is not None

    authed_context.role = ServiceRole.ADMIN
    with pytest.raises(user_errors.UserDeleteError) as exc:
        user_service.delete(context=authed_context, uid=id_to_delete)

    assert exc.type == user_errors.UserDeleteError
    assert exc.value.public_message == expected_error_msg


def test_userservice_delete_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
) -> None:
    id_to_delete = UID()

    @as_result(NotFoundException)
    def mock_delete_by_uid(
        credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> Literal[True]:
        return True

    @as_result(NotFoundException)
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> User:
        return User(email=Faker().email())

    monkeypatch.setattr(user_service.stash, "delete_by_uid", mock_delete_by_uid)
    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    authed_context.role = ServiceRole.ADMIN
    response = user_service.delete(context=authed_context, uid=id_to_delete)
    assert response


def test_userservice_user_verify_key(
    monkeypatch: MonkeyPatch, user_service: UserService, guest_user: User
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Ok:
        return Ok(guest_user)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    response = user_service.user_verify_key(email=guest_user.email).unwrap()
    assert response == guest_user.verify_key


def test_userservice_user_verify_key_invalid_email(
    monkeypatch: MonkeyPatch, user_service: UserService, faker: Faker
) -> None:
    email = faker.email()
    expected_output = f"User {email} not found"

    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> NoReturn:
        raise NotFoundException(public_message=expected_output)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    with pytest.raises(NotFoundException) as exc:
        user_service.user_verify_key(email=email)

    assert exc.type == NotFoundException
    assert exc.value.public_message == expected_output


def test_userservice_admin_verify_key_success(
    monkeypatch: MonkeyPatch, user_service: UserService, worker
) -> None:
    response = user_service.root_verify_key
    assert isinstance(response, SyftVerifyKey)
    assert response == worker.root_client.credentials.verify_key


def test_userservice_register_user_exists(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    worker: Worker,
    guest_create_user: UserCreate,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email) -> User:
        return guest_create_user.to(User)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    expected_error_msg = f"User {guest_create_user.email} already exists"

    # Patch Worker settings to enable signup
    with mock.patch(
        "syft.Worker.settings",
        new_callable=mock.PropertyMock,
        return_value=settings_with_signup_enabled(worker),
    ):
        mock_worker = Worker.named(name="mock-server", db_url="sqlite://")
        server_context = ServerServiceContext(server=mock_worker)

        with pytest.raises(SyftException) as exc:
            user_service.register(server_context, guest_create_user)

        assert exc.type == SyftException
        assert exc.value.public_message == expected_error_msg


def test_userservice_register_error_on_get_email(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    guest_create_user: UserCreate,
    worker: Worker,
) -> None:
    error_msg = "There was an error retrieving data. Contact your admin."

    @as_result(StashException)
    def mock_get_by_email(credentials: SyftVerifyKey, email) -> NoReturn:
        raise StashException

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    # Patch Worker settings to enable signup
    with mock.patch(
        "syft.Worker.settings",
        new_callable=mock.PropertyMock,
        return_value=settings_with_signup_enabled(worker),
    ):
        mock_worker = Worker.named(name="mock-server", db_url="sqlite://")
        server_context = ServerServiceContext(server=mock_worker)

        with pytest.raises(StashException) as exc:
            user_service.register(server_context, guest_create_user)

        assert exc.value.public == error_msg


def test_userservice_register_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    worker: Worker,
    guest_create_user: UserCreate,
    guest_user: User,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> NoReturn:
        raise NotFoundException

    @as_result(StashException)
    def mock_set(*args, **kwargs) -> User:
        return guest_user

    with mock.patch(
        "syft.Worker.settings",
        new_callable=mock.PropertyMock,
        return_value=settings_with_signup_enabled(worker),
    ):
        mock_worker = Worker.named(name="mock-server", db_url="sqlite://")
        server_context = ServerServiceContext(server=mock_worker)

        monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
        monkeypatch.setattr(user_service.stash, "set", mock_set)

        expected_private_key = guest_user.to(UserPrivateKey)
        response = user_service.register(server_context, guest_create_user)

        assert isinstance(response, SyftSuccess)
        user_private_key = response.value
        assert isinstance(user_private_key, UserPrivateKey)
        assert user_private_key == expected_private_key


def test_userservice_register_set_fail(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    worker: Worker,
    guest_create_user: UserCreate,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> NoReturn:
        raise NotFoundException

    @as_result(StashException)
    def mock_set(
        credentials: SyftVerifyKey,
        obj: User,
        add_permissions=None,
        has_permission: bool = False,
    ) -> NoReturn:
        raise StashException

    with mock.patch(
        "syft.Worker.settings",
        new_callable=mock.PropertyMock,
        return_value=settings_with_signup_enabled(worker),
    ):
        mock_worker = Worker.named(name="mock-server", db_url="sqlite://")
        server_context = ServerServiceContext(server=mock_worker)

        monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
        monkeypatch.setattr(user_service.stash, "set", mock_set)

        with pytest.raises(StashException) as exc:
            user_service.register(server_context, guest_create_user)

        assert exc.type is StashException
        assert (
            exc.value.public_message
            == f"Failed to create user {guest_create_user.email}"
        )


def test_userservice_exchange_credentials(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    unauthed_context: UnauthedServiceContext,
    guest_user: User,
) -> None:
    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> User:
        return guest_user

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    expected_user_private_key = guest_user.to(UserPrivateKey)

    response = user_service.exchange_credentials(unauthed_context)
    assert isinstance(response.value, UserPrivateKey)
    assert response.value == expected_user_private_key


def test_userservice_exchange_credentials_invalid_user(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    unauthed_context: UnauthedServiceContext,
    guest_user: User,
) -> None:
    expected_error_msg = f"User {guest_user.email} not found"

    @as_result(NotFoundException)
    def mock_get_by_email(credentials: SyftVerifyKey, email) -> NoReturn:
        raise NotFoundException(public_message=expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    with pytest.raises(NotFoundException) as exc:
        user_service.exchange_credentials(unauthed_context)

    assert exc.type == NotFoundException
    assert exc.value.public_message == expected_error_msg


def test_userservice_exchange_credentials_get_email_fails(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    unauthed_context: UnauthedServiceContext,
) -> None:
    get_by_email_error = "Failed to connect to server."

    @as_result(StashException)
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> NoReturn:
        raise StashException(public_message=get_by_email_error)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    with pytest.raises(StashException) as exc:
        user_service.exchange_credentials(unauthed_context)

    assert exc.type == StashException
    assert exc.value.public_message == get_by_email_error


def test_userservice_update_via_client_with_mixed_args():
    server = orchestra.launch(name="datasite-test", reset=True)

    root_client = server.login(email="info@openmined.org", password="changethis")
    root_client.register(
        name="New user",
        email="new_user@openmined.org",
        password="password",
        password_verify="password",
    )
    assert len(root_client.users.get_all()) == 2

    user_list = root_client.users.search(email="new_user@openmined.org")
    assert len(user_list) == 1

    user = user_list[0]
    assert user.name == "New user"

    root_client.users.update(uid=user.id, name="Updated user name")
    user = root_client.users.search(email="new_user@openmined.org")[0]
    assert user.name == "Updated user name"

    root_client.users.update(user.id, name="User name")
    user = root_client.users.search(email="new_user@openmined.org")[0]
    assert user.name == "User name"

    root_client.users.update(user.id, password="newpassword")
    user_client = root_client.login(
        email="new_user@openmined.org", password="newpassword"
    )
    assert user_client.account.name == "User name"


def test_reset_password():
    server = orchestra.launch(name="datasite-test", reset=True)

    datasite_client = server.login(email="info@openmined.org", password="changethis")
    datasite_client.register(
        email="new_syft_user@openmined.org",
        password="verysecurepassword",
        password_verify="verysecurepassword",
        name="New User",
    )
    guest_client: SyftClient = server.login_as_guest()
    guest_client.forgot_password(email="new_syft_user@openmined.org")
    temp_token = datasite_client.users.request_password_reset(
        datasite_client.notifications[-1].linked_obj.resolve.id
    )
    guest_client.reset_password(token=temp_token, new_password="Password123")
    server.login(email="new_syft_user@openmined.org", password="Password123")


def test_root_cannot_be_deleted():
    server = orchestra.launch(name="datasite-test", reset=True)
    datasite_client = server.login(email="info@openmined.org", password="changethis")

    new_admin_email = "admin@openmined.org"
    new_admin_pass = "changethis2"
    datasite_client.register(
        name="second admin",
        email=new_admin_email,
        password=new_admin_pass,
        password_verify=new_admin_pass,
    )
    # update role
    new_user_id = datasite_client.users.search(email=new_admin_email)[0].id
    datasite_client.users.update(uid=new_user_id, role="admin")

    new_admin_client = server.login(email=new_admin_email, password=new_admin_pass)
    with sy.raises(sy.SyftException):
        new_admin_client.users.delete(datasite_client.account.id)
