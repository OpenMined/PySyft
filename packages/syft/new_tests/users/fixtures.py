# third party
import pytest

# syft absolute
from syft.core.node.new.user import ServiceRole
from syft.core.node.new.user import User
from syft.core.node.new.user import UserCreate
from syft.core.node.new.user import UserPrivateKey
from syft.core.node.new.user import UserSearch
from syft.core.node.new.user import UserView


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
    return user_create


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
    return user_create


@pytest.fixture
def admin_user(admin_create_user) -> User:
    user = admin_create_user.to(User)
    return user


@pytest.fixture
def guest_user(guest_create_user) -> User:
    user = guest_create_user.to(User)
    return user


@pytest.fixture
def admin_view_user(admin_user) -> UserView:
    user_view = admin_user.to(UserView)
    return user_view


@pytest.fixture
def guest_view_user(guest_user) -> UserView:
    user_view = guest_user.to(UserView)
    return user_view


@pytest.fixture
def guest_user_private_key(admin_user) -> UserPrivateKey:
    return UserPrivateKey(email=admin_user.email, signing_key=admin_user.signing_key)


@pytest.fixture
def admin_user_private_key(guest_user) -> UserPrivateKey:
    return UserPrivateKey(email=guest_user.email, signing_key=guest_user.signing_key)


@pytest.fixture
def search_admin(admin_user) -> UserSearch:
    return UserSearch(
        id=admin_user.id,
        name=admin_user.name,
        email=admin_user.email,
        verify_key=admin_user.verify_key,
    )


@pytest.fixture
def search_guest(guest_user) -> UserSearch:
    return UserSearch(
        id=guest_user.id,
        name=guest_user.name,
        email=guest_user.email,
        verify_key=guest_user.verify_key,
    )
