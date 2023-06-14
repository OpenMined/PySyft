# third party
import pytest

# syft absolute
from syft.node.credentials import UserLoginCredentials
from syft.node.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.context import NodeServiceContext
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
from syft.store.document_store import DocumentStore


@pytest.fixture(autouse=True)
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
    return user_create


@pytest.fixture(autouse=True)
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
    return user_create


@pytest.fixture(autouse=True)
def admin_user(admin_create_user) -> User:
    user = admin_create_user.to(User)
    return user


@pytest.fixture(autouse=True)
def guest_user(guest_create_user) -> User:
    user = guest_create_user.to(User)
    return user


@pytest.fixture(autouse=True)
def admin_view_user(admin_user) -> UserView:
    user_view = admin_user.to(UserView)
    return user_view


@pytest.fixture(autouse=True)
def guest_view_user(guest_user) -> UserView:
    user_view = guest_user.to(UserView)
    return user_view


def admin_user_private_key(admin_user) -> UserPrivateKey:
    return UserPrivateKey(
        email=admin_user.email,
        signing_key=admin_user.signing_key,
        role=ServiceRole.DATA_OWNER,
    )


@pytest.fixture
def guest_user_private_key(guest_user) -> UserPrivateKey:
    return UserPrivateKey(
        email=guest_user.email,
        signing_key=guest_user.signing_key,
        role=ServiceRole.GUEST,
    )


@pytest.fixture(autouse=True)
def update_user(faker) -> UserSearch:
    return UserUpdate(
        name=faker.name(),
        email=faker.email(),
    )


@pytest.fixture(autouse=True)
def guest_user_search(guest_user) -> UserSearch:
    return UserSearch(
        name=guest_user.name, email=guest_user.email, verify_key=guest_user.verify_key
    )


@pytest.fixture(autouse=True)
def user_stash(document_store: DocumentStore) -> UserStash:
    return UserStash(store=document_store)


@pytest.fixture
def user_service(document_store: DocumentStore):
    return UserService(store=document_store)


@pytest.fixture
def authed_context(admin_user: User, worker: Worker) -> AuthedServiceContext:
    return AuthedServiceContext(credentials=admin_user.verify_key, node=worker)


@pytest.fixture
def node_context(worker: Worker) -> NodeServiceContext:
    return NodeServiceContext(node=worker)


@pytest.fixture
def unauthed_context(
    guest_create_user: UserCreate, worker: Worker
) -> UnauthedServiceContext:
    login_credentials = UserLoginCredentials(
        email=guest_create_user.email, password=guest_create_user.password
    )
    return UnauthedServiceContext(login_credentials=login_credentials, node=worker)
