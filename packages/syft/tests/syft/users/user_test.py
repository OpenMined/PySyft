# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.client.api import SyftAPICall
from syft.client.domain_client import DomainClient
from syft.node.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.user.user import ServiceRole
from syft.service.user.user import UserCreate
from syft.service.user.user import UserUpdate
from syft.service.user.user import UserView

GUEST_ROLES = [ServiceRole.GUEST]
DS_ROLES = [ServiceRole.GUEST, ServiceRole.DATA_SCIENTIST]
DO_ROLES = [ServiceRole.GUEST, ServiceRole.DATA_SCIENTIST, ServiceRole.DATA_OWNER]
ADMIN_ROLES = [
    ServiceRole.GUEST,
    ServiceRole.DATA_SCIENTIST,
    ServiceRole.DATA_OWNER,
    ServiceRole.ADMIN,
]


def get_users(worker):
    return worker.get_service("UserService").get_all(
        AuthedServiceContext(node=worker, credentials=worker.signing_key.verify_key)
    )


def get_mock_client(root_client, role):
    worker = root_client.api.connection.node
    client = worker.guest_client
    mail = Faker().email()
    name = Faker().name()
    password = "pw"
    assert root_client.register(name=name, email=mail, password=password)
    user_id = [u for u in get_users(worker) if u.email == mail][0].id
    assert worker.root_client.api.services.user.update(
        user_id, UserUpdate(user_id=user_id, role=role)
    )
    client.login(email=mail, password=password)
    client._fetch_api(client.credentials)
    # hacky, but useful for testing: patch user id and role on client
    client.user_id = user_id
    client.role = role
    return client


def manually_call_service(worker, client, service, args=None, kwargs=None):
    # we want to use this function because just making the call will hit the client side permissions,
    # while we mostly want to validate the server side permissions.
    args = args if args is not None else []
    kwargs = kwargs if kwargs is not None else {}
    api_call = SyftAPICall(node_uid=worker.id, path=service, args=args, kwargs=kwargs)
    signed_call = api_call.sign(client.api.signing_key)
    signed_result = client.api.connection.make_call(signed_call)
    result = signed_result.message.data
    return result


@pytest.fixture
def guest_client(worker):
    return get_mock_client(worker.root_client, ServiceRole.GUEST)


@pytest.fixture
def ds_client(worker):
    return get_mock_client(worker.root_client, ServiceRole.DATA_SCIENTIST)


@pytest.fixture
def do_client(worker):
    return get_mock_client(worker.root_client, ServiceRole.DATA_OWNER)


# this shadows the normal conftests.py/root_client, but I am experiencing a lot of problems
# with that fixture
@pytest.fixture
def root_client(worker):
    return get_mock_client(worker.root_client, ServiceRole.DATA_OWNER)


def test_read_user(worker, root_client, do_client, ds_client, guest_client):
    for client in [ds_client, guest_client]:
        assert not manually_call_service(worker, client, "user.get_all")

    for client in [do_client, root_client]:
        assert manually_call_service(worker, client, "user.get_all")


def test_read_returns_view(root_client):
    # Test reading returns userview (and not real user), this wasnt the case, adding this as a sanity check
    users = root_client.api.services.user
    assert len(list(users))
    for _ in users:
        # check that result has no sensitive information
        assert isinstance(root_client.api.services.user[0], UserView)


def test_user_create(worker, do_client, guest_client, ds_client, root_client):
    for client in [ds_client, guest_client]:
        assert not manually_call_service(worker, client, "user.create")
    for client in [do_client, root_client]:
        res = manually_call_service(
            worker,
            client,
            "user.create",
            args=[
                UserCreate(
                    email=Faker().email(), name="z", password="pw", password_verify="pw"
                )
            ],
        )
        assert isinstance(res, UserView)


def test_user_delete(do_client, guest_client, ds_client, worker, root_client):
    # admins can delete lower users
    clients = [get_mock_client(root_client, role) for role in DO_ROLES]
    for c in clients:
        assert worker.root_client.api.services.user.delete(c.user_id)

    # admins can delete other admins
    assert worker.root_client.api.services.user.delete(
        get_mock_client(root_client, ServiceRole.ADMIN).user_id
    )
    admin_client3 = get_mock_client(root_client, ServiceRole.ADMIN)

    # admins can delete themselves
    assert admin_client3.api.services.user.delete(admin_client3.user_id)

    # DOs can delete lower roles
    clients = [get_mock_client(root_client, role) for role in DS_ROLES]
    for c in clients:
        assert do_client.api.services.user.delete(c.user_id)
    # but not higher or same roles
    clients = [
        get_mock_client(root_client, role)
        for role in [ServiceRole.DATA_OWNER, ServiceRole.ADMIN]
    ]
    for c in clients:
        assert not do_client.api.services.user.delete(c.user_id)

    # DS cannot delete anything
    clients = [get_mock_client(root_client, role) for role in ADMIN_ROLES]
    for c in clients:
        assert not ds_client.api.services.user.delete(c.user_id)

    # Guests cannot delete anything
    clients = [get_mock_client(root_client, role) for role in ADMIN_ROLES]
    for c in clients:
        assert not guest_client.api.services.user.delete(c.user_id)


def test_user_update_roles(do_client, guest_client, ds_client, root_client, worker):
    # admins can update the roles of lower roles
    clients = [get_mock_client(root_client, role) for role in DO_ROLES]
    for c in clients:
        assert worker.root_client.api.services.user.update(
            c.user_id, UserUpdate(role=ServiceRole.ADMIN)
        )

    # DOs can update the roles of lower roles
    clients = [get_mock_client(root_client, role) for role in DS_ROLES]
    for c in clients:
        assert do_client.api.services.user.update(
            c.user_id, UserUpdate(role=ServiceRole.DATA_SCIENTIST)
        )

    clients = [get_mock_client(root_client, role) for role in ADMIN_ROLES]

    # DOs cannot update roles to greater than / equal to own role
    for c in clients:
        for target_role in [ServiceRole.DATA_OWNER, ServiceRole.ADMIN]:
            assert not do_client.api.services.user.update(
                c.user_id, UserUpdate(role=target_role)
            )

    # DOs cannot downgrade higher roles to lower levels
    clients = [
        get_mock_client(root_client, role)
        for role in [ServiceRole.ADMIN, ServiceRole.DATA_OWNER]
    ]
    for c in clients:
        for target_role in DO_ROLES:
            if target_role < c.role:
                assert not do_client.api.services.user.update(
                    c.user_id, UserUpdate(role=target_role)
                )

    # DSs cannot update any roles
    clients = [get_mock_client(root_client, role) for role in ADMIN_ROLES]
    for c in clients:
        for target_role in ADMIN_ROLES:
            assert not ds_client.api.services.user.update(
                c.user_id, UserUpdate(role=target_role)
            )

    # Guests cannot update any roles
    clients = [get_mock_client(root_client, role) for role in ADMIN_ROLES]
    for c in clients:
        for target_role in ADMIN_ROLES:
            assert not guest_client.api.services.user.update(
                c.user_id, UserUpdate(role=target_role)
            )


def test_user_update(root_client):
    executing_clients = [get_mock_client(root_client, role) for role in ADMIN_ROLES]
    target_clients = [get_mock_client(root_client, role) for role in ADMIN_ROLES]

    for executing_client in executing_clients:
        for target_client in target_clients:
            if executing_client.role != ServiceRole.ADMIN:
                assert not executing_client.api.services.user.update(
                    target_client.user_id, UserUpdate(name="abc")
                )
            else:
                assert executing_client.api.services.user.update(
                    target_client.user_id, UserUpdate(name="abc")
                )

        # you can update yourself
        assert executing_client.api.services.user.update(
            executing_client.user_id, UserUpdate(name=Faker().name())
        )


def test_user_view_set_password(worker: Worker, root_client: DomainClient) -> None:
    root_client.me.set_pw("123")
    email = root_client.me.email
    # log in again with the wrong password
    root_client_c = worker.root_client.login(email=email, password="1234")
    assert isinstance(root_client_c, sy.SyftError)
    # log in again with the right password
    root_client_b = worker.root_client.login(email=email, password="123")
    assert root_client_b.me == root_client.me
