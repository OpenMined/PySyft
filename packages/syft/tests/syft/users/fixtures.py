# third party
import pytest

# syft absolute
from syft.server.credentials import UserLoginCredentials
from syft.server.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.context import ServerServiceContext
from syft.service.context import UnauthedServiceContext
from syft.service.user.user import User
from syft.service.user.user import UserCreate
from syft.service.user.user import UserPrivateKey
from syft.service.user.user import UserSearch
from syft.service.user.user import UserUpdate
from syft.service.user.user import UserView
from syft.service.user.user_roles import ServiceRole
from syft.service.user.user_service import UserService
from syft.service.user.user_stash import UserStash


@pytest.fixture
def admin_create_user(faker) -> UserCreate:
    password = faker.password()
    user_create = UserCreate(
        email=faker.company_email(),
        name=faker.name(),
        role=ServiceRole.ADMIN,
        password=password,
        password_verify=password,
        institution=faker.company(),
        website=faker.url(),
    )
    yield user_create


@pytest.fixture
def guest_create_user(faker) -> UserCreate:
    password = faker.password()
    user_create = UserCreate(
        email=faker.company_email(),
        name=faker.name(),
        role=ServiceRole.GUEST,
        password=password,
        password_verify=password,
        institution=faker.company(),
        website=faker.url(),
    )
    yield user_create


@pytest.fixture
def admin_user(admin_create_user) -> User:
    user = admin_create_user.to(User)
    yield user


@pytest.fixture
def guest_user(guest_create_user) -> User:
    user = guest_create_user.to(User)
    yield user


@pytest.fixture
def admin_view_user(admin_user) -> UserView:
    user_view = admin_user.to(UserView)
    yield user_view


@pytest.fixture
def guest_view_user(guest_user) -> UserView:
    user_view = guest_user.to(UserView)
    yield user_view


@pytest.fixture
def admin_user_private_key(admin_user) -> UserPrivateKey:
    yield UserPrivateKey(
        email=admin_user.email,
        signing_key=admin_user.signing_key,
        role=ServiceRole.DATA_OWNER,
    )


@pytest.fixture
def guest_user_private_key(guest_user) -> UserPrivateKey:
    yield UserPrivateKey(
        email=guest_user.email,
        signing_key=guest_user.signing_key,
        role=ServiceRole.GUEST,
    )


@pytest.fixture
def update_user(faker) -> UserSearch:
    yield UserUpdate(
        name=faker.name(),
        email=faker.email(),
    )


@pytest.fixture
def guest_user_search(guest_user) -> UserSearch:
    yield UserSearch(
        name=guest_user.name, email=guest_user.email, verify_key=guest_user.verify_key
    )


@pytest.fixture
def user_stash(document_store) -> UserStash:
    yield UserStash(store=document_store)


@pytest.fixture
def user_service(document_store):
    yield UserService(store=document_store)


@pytest.fixture
def authed_context(admin_user: User, worker: Worker) -> AuthedServiceContext:
    yield AuthedServiceContext(credentials=admin_user.verify_key, server=worker)


@pytest.fixture
def server_context(worker: Worker) -> ServerServiceContext:
    yield ServerServiceContext(server=worker)


@pytest.fixture
def unauthed_context(
    guest_create_user: UserCreate, worker: Worker
) -> UnauthedServiceContext:
    login_credentials = UserLoginCredentials(
        email=guest_create_user.email, password=guest_create_user.password
    )
    yield UnauthedServiceContext(login_credentials=login_credentials, server=worker)
