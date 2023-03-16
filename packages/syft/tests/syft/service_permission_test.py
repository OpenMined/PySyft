# third party
import pytest

# syft absolute
import syft as sy
from syft import SyftError
from syft.core.node.new.api import SyftAPICall
from syft.core.node.new.user import UserUpdate


@pytest.fixture
def worker_domain_guest():
    worker = sy.Worker.named("test-domain-service-permissions", processes=0, reset=True)
    domain_client = worker.root_client
    guest_domain_client = domain_client.guest()
    guest_domain_client.register(
        name="Alice", email="alice@caltech.edu", password="abc123"
    )
    return [worker, domain_client, guest_domain_client]


def test_call_service_syftapi_with_permission(worker_domain_guest):
    worker, _, guest_domain_client = worker_domain_guest
    user_id = worker.document_store.partitions["User"].all().value[-1].id
    res = guest_domain_client.api.services.user.update(
        user_id, UserUpdate(user_id=user_id, name="")
    )
    assert res


# this throws an AttributeError, maybe we want something more clear?
def test_call_service_syftapi_no_permission(worker_domain_guest):
    _, __, guest_domain_client = worker_domain_guest
    with pytest.raises(AttributeError):
        guest_domain_client.api.services.user.get_all()


def test_directly_call_service_with_permission(worker_domain_guest):
    worker, _, guest_domain_client = worker_domain_guest
    user_id = worker.document_store.partitions["User"].all().value[-1].id
    api_call = SyftAPICall(
        node_uid=worker.id,
        path="user.update",
        args=[user_id, UserUpdate(name="")],
        kwargs={},
    )
    signed_call = api_call.sign(guest_domain_client.api.signing_key)
    signed_result = guest_domain_client.api.connection.make_call(signed_call)
    result = signed_result.message.data
    assert result


def test_directly_call_service_no_permission(worker_domain_guest):
    worker, _, guest_domain_client = worker_domain_guest
    api_call = SyftAPICall(node_uid=worker.id, path="user.get_all", args=[], kwargs={})
    signed_call = api_call.sign(guest_domain_client.api.signing_key)
    signed_result = guest_domain_client.api.connection.make_call(signed_call)
    result = signed_result.message.data
    assert isinstance(result, SyftError)
