# stdlib
import os
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

# third party
from faker import Faker
import pytest
from pytest import MonkeyPatch
from result import Err
from result import Ok

# syft absolute
from syft.node.credentials import SyftVerifyKey
from syft.service.context import AuthedServiceContext
from syft.service.context import NodeServiceContext
from syft.service.context import UnauthedServiceContext
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.user.user import User
from syft.service.user.user import UserCreate
from syft.service.user.user import UserPrivateKey
from syft.service.user.user import UserUpdate
from syft.service.user.user import UserView
from syft.service.user.user_roles import ServiceRole
from syft.service.user.user_service import UserService
from syft.types.uid import UID


@pytest.fixture()
def node_with_signup_enabled(worker) -> Type:
    mock_metadata = worker.metadata
    mock_metadata.signup_enabled = True

    class NodewithSignupEnabled:
        metadata = mock_metadata

    return NodewithSignupEnabled


def test_userservice_create_when_user_exists(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_create_user: UserCreate,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Ok:
        return Ok(guest_create_user.to(User))

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    response = user_service.create(authed_context, guest_create_user)
    assert isinstance(response, SyftError)
    expected_error_message = (
        f"User already exists with email: {guest_create_user.email}"
    )
    assert expected_error_message == response.message


def test_userservice_create_error_on_get_by_email(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_create_user: UserCreate,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Err:
        return Err(f"No user exists with given email: {email}")

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    response = user_service.create(authed_context, guest_create_user)
    assert isinstance(response, SyftError)
    expected_error_message = mock_get_by_email(None, guest_create_user.email).err()
    assert response.message == expected_error_message


def test_userservice_create_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_create_user: UserCreate,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Ok:
        return Ok(None)

    expected_user = guest_create_user.to(User)
    expected_output = expected_user.to(UserView)

    def mock_set(
        credentials: SyftVerifyKey,
        user: User,
        has_permission: bool = False,
        add_permissions=None,
    ) -> Ok:
        return Ok(expected_user)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    monkeypatch.setattr(user_service.stash, "set", mock_set)
    response = user_service.create(authed_context, guest_create_user)
    assert isinstance(response, UserView)
    assert response.to_dict() == expected_output.to_dict()


def test_userservice_create_error_on_set(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_create_user: UserCreate,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Ok:
        return Ok(None)

    expected_error_msg = "Failed to set user."

    def mock_set(
        credentials: SyftVerifyKey,
        user: User,
        has_permission: bool = False,
        add_permissions=None,
    ) -> Err:
        return Err(expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    monkeypatch.setattr(user_service.stash, "set", mock_set)
    response = user_service.create(authed_context, guest_create_user)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_view_error_on_get_by_uid(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
) -> None:
    uid_to_view = UID()
    expected_error_msg = f"Failed to get uid: {uid_to_view}"

    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> Err:
        return Err(expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)
    response = user_service.view(authed_context, uid_to_view)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_view_user_not_exists(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
) -> None:
    uid_to_view = UID()
    expected_error_msg = f"No user exists for given: {uid_to_view}"

    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> Ok:
        return Ok(None)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)
    response = user_service.view(authed_context, uid_to_view)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_view_user_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
) -> None:
    uid_to_view = guest_user.id
    expected_output = guest_user.to(UserView)

    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> Ok:
        return Ok(guest_user)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)
    response = user_service.view(authed_context, uid_to_view)
    assert isinstance(response, UserView)
    assert response == expected_output


def test_userservice_get_all_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
    admin_user: User,
) -> None:
    mock_get_all_output = [guest_user, admin_user]
    expected_output = [x.to(UserView) for x in mock_get_all_output]

    def mock_get_all(credentials: SyftVerifyKey) -> Ok:
        return Ok(mock_get_all_output)

    monkeypatch.setattr(user_service.stash, "get_all", mock_get_all)
    response = user_service.get_all(authed_context)
    assert isinstance(response, List)
    assert len(response) == len(expected_output)
    assert response == expected_output


def test_userservice_get_all_error(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_output_msg = "No users exists"

    def mock_get_all(credentials: SyftVerifyKey) -> Err:
        return Err("")

    monkeypatch.setattr(user_service.stash, "get_all", mock_get_all)
    response = user_service.get_all(authed_context)
    assert isinstance(response, SyftError)
    assert response.message == expected_output_msg


def test_userservice_search(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
) -> None:
    def mock_find_all(credentials: SyftVerifyKey, **kwargs) -> Union[Ok, Err]:
        for key, _ in kwargs.items():
            if hasattr(guest_user, key):
                return Ok([guest_user])
            return Err("Invalid kwargs")

    monkeypatch.setattr(user_service.stash, "find_all", mock_find_all)

    expected_output = [guest_user.to(UserView)]

    # Search via id
    response = user_service.search(authed_context, id=guest_user.id)
    assert isinstance(response, List)
    assert response == expected_output

    # Search via email
    response = user_service.search(authed_context, email=guest_user.email)
    assert isinstance(response, List)
    assert response == expected_output

    # Search via name
    response = user_service.search(authed_context, name=guest_user.name)
    assert isinstance(response, List)
    assert response == expected_output

    # Search via verify_key
    response = user_service.search(
        authed_context,
        verify_key=guest_user.verify_key,
    )
    assert isinstance(response, List)
    assert response == expected_output

    # Search via multiple kwargs
    response = user_service.search(
        authed_context, name=guest_user.name, email=guest_user.email
    )
    assert isinstance(response, List)
    assert response == expected_output


def test_userservice_search_with_invalid_kwargs(
    user_service: UserService, authed_context: AuthedServiceContext
) -> None:
    # Search with invalid kwargs
    response = user_service.search(authed_context, role=ServiceRole.GUEST)
    assert isinstance(response, SyftError)
    assert "Invalid Search parameters" in response.message


def test_userservice_update_get_by_uid_fails(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    update_user: UserUpdate,
) -> None:
    random_uid = UID()
    get_by_uid_err_msg = "Invalid UID"
    expected_error_msg = (
        f"Failed to find user with UID: {random_uid}. Error: {get_by_uid_err_msg}"
    )

    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> Err:
        return Err(get_by_uid_err_msg)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    response = user_service.update(
        authed_context, uid=random_uid, user_update=update_user
    )
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_update_no_user_exists(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    update_user: UserUpdate,
) -> None:
    random_uid = UID()
    expected_error_msg = f"No user exists for given UID: {random_uid}"

    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> Ok:
        return Ok(None)

    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    response = user_service.update(
        authed_context, uid=random_uid, user_update=update_user
    )
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_update_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
    update_user: UserUpdate,
) -> None:
    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> Ok:
        return Ok(guest_user)

    def mock_update(credentials: SyftVerifyKey, user: User, has_permission: bool) -> Ok:
        guest_user.name = update_user.name
        guest_user.email = update_user.email
        return Ok(guest_user)

    monkeypatch.setattr(user_service.stash, "update", mock_update)
    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)
    authed_context.role = ServiceRole.ADMIN

    resultant_user = user_service.update(
        authed_context, uid=guest_user.id, user_update=update_user
    )
    assert isinstance(resultant_user, UserView)
    assert resultant_user.email == update_user.email
    assert resultant_user.name == update_user.name


def test_userservice_update_fails(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
    guest_user: User,
    update_user: UserUpdate,
) -> None:
    update_error_msg = "Failed to reach server."
    expected_error_msg = (
        f"Failed to update user with UID: {guest_user.id}. Error: {update_error_msg}"
    )

    def mock_get_by_uid(credentials: SyftVerifyKey, uid: UID) -> Ok:
        return Ok(guest_user)

    def mock_update(credentials: SyftVerifyKey, user, has_permission: bool) -> Err:
        return Err(update_error_msg)

    authed_context.role = ServiceRole.ADMIN

    monkeypatch.setattr(user_service.stash, "update", mock_update)
    monkeypatch.setattr(user_service.stash, "get_by_uid", mock_get_by_uid)

    response = user_service.update(
        authed_context, uid=guest_user.id, user_update=update_user
    )
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_delete_failure(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
) -> None:
    id_to_delete = UID()
    expected_error_msg = f"No user exists for given id: {id_to_delete}"

    def mock_delete_by_uid(
        credentials: SyftVerifyKey, uid: UID, has_permission=False
    ) -> Err:
        return Err(expected_error_msg)

    monkeypatch.setattr(user_service.stash, "delete_by_uid", mock_delete_by_uid)

    response = user_service.delete(context=authed_context, uid=id_to_delete)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_delete_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    authed_context: AuthedServiceContext,
) -> None:
    id_to_delete = UID()
    expected_output = SyftSuccess(message=f"ID: {id_to_delete} deleted")

    def mock_delete_by_uid(
        credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> Ok:
        return Ok(expected_output)

    def mock_get_target_object(credentials: SyftVerifyKey, uid):
        return User(email=Faker().email())

    monkeypatch.setattr(user_service.stash, "delete_by_uid", mock_delete_by_uid)
    monkeypatch.setattr(user_service, "get_target_object", mock_get_target_object)
    authed_context.role = ServiceRole.ADMIN

    response = user_service.delete(context=authed_context, uid=id_to_delete)
    assert isinstance(response, SyftSuccess)
    assert response == expected_output


def test_userservice_user_verify_key(
    monkeypatch: MonkeyPatch, user_service: UserService, guest_user: User
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Ok:
        return Ok(guest_user)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    response = user_service.user_verify_key(email=guest_user.email)
    assert response == guest_user.verify_key


def test_userservice_user_verify_key_invalid_email(
    monkeypatch: MonkeyPatch, user_service: UserService, faker: Faker
) -> None:
    email = faker.email()
    expected_output = SyftError(message=f"No user with email: {email}")

    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Err:
        return Err("No user found")

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)

    response = user_service.user_verify_key(email=email)
    assert response == expected_output


def test_userservice_admin_verify_key_error(
    monkeypatch: MonkeyPatch, user_service: UserService
) -> None:
    expected_output = "failed to get admin verify_key"

    def mock_admin_verify_key() -> Err:
        return Err(expected_output)

    monkeypatch.setattr(user_service.stash, "admin_verify_key", mock_admin_verify_key)

    response = user_service.admin_verify_key()
    assert isinstance(response, SyftError)
    assert response.message == expected_output


def test_userservice_admin_verify_key_success(
    monkeypatch: MonkeyPatch, user_service: UserService, worker
) -> None:
    response = user_service.admin_verify_key()
    assert isinstance(response, SyftVerifyKey)
    assert response == worker.root_client.credentials.verify_key


def test_userservice_register_user_exists(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    node_context: NodeServiceContext,
    guest_create_user: UserCreate,
    node_with_signup_enabled: Type,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email):
        return Ok(guest_create_user)

    monkeypatch.setattr(node_context.node, "__class__", node_with_signup_enabled)
    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    expected_error_msg = f"User already exists with email: {guest_create_user.email}"

    response = user_service.register(node_context, guest_create_user)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_register_error_on_get_email(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    node_context: NodeServiceContext,
    guest_create_user: UserCreate,
    node_with_signup_enabled: Type,
) -> None:
    expected_error_msg = "Failed to get email"

    def mock_get_by_email(credentials: SyftVerifyKey, email):
        return Err(expected_error_msg)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    monkeypatch.setattr(node_context.node, "__class__", node_with_signup_enabled)

    response = user_service.register(node_context, guest_create_user)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_register_success(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    node_context: NodeServiceContext,
    guest_create_user: UserCreate,
    guest_user: User,
    node_with_signup_enabled: Type,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Ok:
        return Ok(None)

    def mock_set(
        credentials: SyftVerifyKey,
        user: str,
        has_permission: bool = False,
        add_permissions=None,
    ) -> Ok:
        return Ok(guest_user)

    monkeypatch.setattr(node_context.node, "__class__", node_with_signup_enabled)
    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    monkeypatch.setattr(user_service.stash, "set", mock_set)

    expected_msg = "User successfully registered!"
    expected_private_key = guest_user.to(UserPrivateKey)

    response = user_service.register(node_context, guest_create_user)
    assert isinstance(response, Tuple)

    syft_success_response, user_private_key = response
    assert isinstance(syft_success_response, SyftSuccess)
    assert syft_success_response.message == expected_msg

    assert isinstance(user_private_key, UserPrivateKey)
    assert user_private_key == expected_private_key


def test_userservice_register_set_fail(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    node_context: NodeServiceContext,
    guest_create_user: UserCreate,
    node_with_signup_enabled: Type,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Ok:
        return Ok(None)

    expected_error_msg = "Failed to connect to server."

    def mock_set(
        credentials: SyftVerifyKey,
        user: User,
        add_permissions=None,
        has_permission: bool = False,
    ) -> Err:
        return Err(expected_error_msg)

    monkeypatch.setattr(node_context.node, "__class__", node_with_signup_enabled)
    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    monkeypatch.setattr(user_service.stash, "set", mock_set)

    response = user_service.register(node_context, guest_create_user)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_exchange_credentials(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    unauthed_context: UnauthedServiceContext,
    guest_user: User,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Ok:
        return Ok(guest_user)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    expected_user_private_key = guest_user.to(UserPrivateKey)

    response = user_service.exchange_credentials(unauthed_context)
    assert isinstance(response, UserPrivateKey)
    assert response == expected_user_private_key


def test_userservice_exchange_credentials_invalid_user(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    unauthed_context: UnauthedServiceContext,
    guest_user: User,
) -> None:
    def mock_get_by_email(credentials: SyftVerifyKey, email):
        return Ok(None)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    expected_error_msg = (
        f"No user exists with {guest_user.email} and supplied password."
    )

    response = user_service.exchange_credentials(unauthed_context)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_exchange_credentials_get_email_fails(
    monkeypatch: MonkeyPatch,
    user_service: UserService,
    unauthed_context: UnauthedServiceContext,
    guest_user: User,
) -> None:
    get_by_email_error = "Failed to connect to server."

    def mock_get_by_email(credentials: SyftVerifyKey, email: str) -> Err:
        return Err(get_by_email_error)

    monkeypatch.setattr(user_service.stash, "get_by_email", mock_get_by_email)
    expected_error_msg = f"Failed to retrieve user with {guest_user.email} with error: {get_by_email_error}"

    response = user_service.exchange_credentials(unauthed_context)
    assert isinstance(response, SyftError)
    assert response.message == expected_error_msg


def test_userservice_toggle_registration(
    faker, guest_domain_client, root_domain_client
) -> None:
    os.environ.setdefault("ENABLE_SIGNUP", "False")
    email1 = faker.email()
    email2 = faker.email()
    response_1 = root_domain_client.register(
        email=email1, password="joker123", name="Joker"
    )
    assert isinstance(response_1, SyftSuccess)
    # by default, the guest client can't register new user
    response_2 = guest_domain_client.register(
        email=email2, password="harley123", name="Harley"
    )
    assert isinstance(response_2, SyftError)

    assert any([user.email == email1 for user in root_domain_client.users])

    # only after the root client enable other users to signup, they can
    root_domain_client.users.toggle_signup(enable=True)
    response_3 = guest_domain_client.register(
        email=email2, password="harley123", name="Harley"
    )
    assert isinstance(response_3, SyftSuccess)

    assert any([user.email == email2 for user in root_domain_client.users])

    # if the root client turn off the signup option, guest users can't register anymore
    root_domain_client.users.toggle_signup(enable=False)
    response_4 = guest_domain_client.register(
        email="batman@test.com", password="batman123", name="Batman"
    )
    assert isinstance(response_4, SyftError)
