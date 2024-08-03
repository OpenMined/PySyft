# third party
import pytest

# syft absolute
from syft.server.credentials import UserLoginCredentials
from syft.server.worker import Worker
from syft.service.context import (
    AuthedServiceContext,
    ServerServiceContext,
    UnauthedServiceContext,
)
from syft.service.user.user import (
    User,
    UserCreate,
    UserPrivateKey,
    UserSearch,
    UserUpdate,
    UserView,
)
from syft.service.user.user_roles import ServiceRole
from syft.service.user.user_service import UserService
from syft.service.user.user_stash import UserStash
from syft.store.document_store import DocumentStore


@pytest.fixture()
def admin_create_user(faker) -> UserCreate:
    password = faker.password()
    return UserCreate(
        email=faker.company_email(),
        name=faker.name(),
        role=ServiceRole.ADMIN,
        password=password,
        password_verify=password,
        institution=faker.company(),
        website=faker.url(),
    )


@pytest.fixture()
def guest_create_user(faker) -> UserCreate:
    password = faker.password()
    return UserCreate(
        email=faker.company_email(),
        name=faker.name(),
        role=ServiceRole.GUEST,
        password=password,
        password_verify=password,
        institution=faker.company(),
        website=faker.url(),
    )


@pytest.fixture()
def admin_user(admin_create_user) -> User:
    return admin_create_user.to(User)


@pytest.fixture()
def guest_user(guest_create_user) -> User:
    return guest_create_user.to(User)


@pytest.fixture()
def admin_view_user(admin_user) -> UserView:
    return admin_user.to(UserView)


@pytest.fixture()
def guest_view_user(guest_user) -> UserView:
    return guest_user.to(UserView)


@pytest.fixture()
def admin_user_private_key(admin_user) -> UserPrivateKey:
    return UserPrivateKey(
        email=admin_user.email,
        signing_key=admin_user.signing_key,
        role=ServiceRole.DATA_OWNER,
    )


@pytest.fixture()
def guest_user_private_key(guest_user) -> UserPrivateKey:
    return UserPrivateKey(
        email=guest_user.email,
        signing_key=guest_user.signing_key,
        role=ServiceRole.GUEST,
    )


@pytest.fixture()
def update_user(faker) -> UserSearch:
    return UserUpdate(
        name=faker.name(),
        email=faker.email(),
    )


@pytest.fixture()
def guest_user_search(guest_user) -> UserSearch:
    return UserSearch(
        name=guest_user.name, email=guest_user.email, verify_key=guest_user.verify_key,
    )


@pytest.fixture()
def user_stash(document_store: DocumentStore) -> UserStash:
    return UserStash(store=document_store)


@pytest.fixture()
def user_service(document_store: DocumentStore):
    return UserService(store=document_store)


@pytest.fixture()
def authed_context(admin_user: User, worker: Worker) -> AuthedServiceContext:
    return AuthedServiceContext(credentials=admin_user.verify_key, server=worker)


@pytest.fixture()
def server_context(worker: Worker) -> ServerServiceContext:
    return ServerServiceContext(server=worker)


@pytest.fixture()
def unauthed_context(
    guest_create_user: UserCreate, worker: Worker,
) -> UnauthedServiceContext:
    login_credentials = UserLoginCredentials(
        email=guest_create_user.email, password=guest_create_user.password,
    )
    return UnauthedServiceContext(login_credentials=login_credentials, server=worker)
