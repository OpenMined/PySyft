# something here
# third party
import pytest

# syft absolute
from syft.core.node.new.request_stash import RequestStash


@pytest.fixture
def request_stash(document_store) -> RequestStash:
    return RequestStash(store=document_store)
