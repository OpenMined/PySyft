# something here
# third party
import pytest

# syft absolute
from syft.client.client import SyftClient
from syft.node.credentials import SyftVerifyKey
from syft.node.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.request.request_stash import RequestStash
from syft.store.document_store import DocumentStore


@pytest.fixture
def request_stash(document_store: DocumentStore) -> RequestStash:
    yield RequestStash(store=document_store)


@pytest.fixture
def authed_context_guest_datasite_client(
    guest_datasite_client: SyftClient, worker: Worker
) -> AuthedServiceContext:
    verify_key: SyftVerifyKey = guest_datasite_client.credentials.verify_key
    yield AuthedServiceContext(credentials=verify_key, node=worker)
