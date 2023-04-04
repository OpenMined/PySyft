# something here
# third party
import pytest

# syft absolute
from syft.core.node.new.client import SyftClient
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.document_store import DocumentStore
from syft.core.node.new.request_stash import RequestStash
from syft.core.node.new.user import User
from syft.core.node.worker import Worker


@pytest.fixture
def request_stash(document_store: DocumentStore) -> RequestStash:
    return RequestStash(store=document_store)


@pytest.fixture
def authed_context(admin_user: User, worker: Worker) -> AuthedServiceContext:
    return AuthedServiceContext(credentials=admin_user.verify_key, node=worker)


@pytest.fixture
def authed_context_guest_domain_client(
    guest_domain_client: SyftClient, worker: Worker
) -> AuthedServiceContext:
    verify_key: SyftVerifyKey = guest_domain_client.credentials.verify_key
    return AuthedServiceContext(credentials=verify_key, node=worker)
