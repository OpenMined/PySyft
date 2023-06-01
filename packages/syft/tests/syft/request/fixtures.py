# something here
# third party
import pytest

# syft absolute
from syft.client.client import SyftClient
from syft.node.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.request.request_stash import RequestStash
from syft.store.document_store import DocumentStore
from syft.types.credentials import SyftVerifyKey


@pytest.fixture
def request_stash(document_store: DocumentStore) -> RequestStash:
    return RequestStash(store=document_store)


@pytest.fixture
def authed_context_guest_domain_client(
    guest_domain_client: SyftClient, worker: Worker
) -> AuthedServiceContext:
    verify_key: SyftVerifyKey = guest_domain_client.credentials.verify_key
    return AuthedServiceContext(credentials=verify_key, node=worker)
