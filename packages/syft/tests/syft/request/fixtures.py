# something here
# third party
import pytest

# syft absolute
from syft.client.client import SyftClient
from syft.server.credentials import SyftVerifyKey
from syft.server.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.request.request_service import RequestService
from syft.service.request.request_stash import RequestStash
from syft.store.document_store import DocumentStore


@pytest.fixture()
def request_stash(document_store: DocumentStore) -> RequestStash:
    return RequestStash(store=document_store)


@pytest.fixture()
def authed_context_guest_datasite_client(
    guest_datasite_client: SyftClient, worker: Worker,
) -> AuthedServiceContext:
    verify_key: SyftVerifyKey = guest_datasite_client.credentials.verify_key
    return AuthedServiceContext(credentials=verify_key, server=worker)


@pytest.fixture()
def request_service(document_store: DocumentStore):
    return RequestService(store=document_store)
