# third party
import pytest

# syft absolute
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.project_stash import ProjectStash
from syft.core.node.new.document_store import DocumentStore, PartitionKey


@pytest.fixture(autouse=True)
def project_stash(document_store: DocumentStore) -> ProjectStash:
    return ProjectStash(store=document_store)

ProjectUserVerifyKeyPartitionKey = PartitionKey(
    key="user_verify_key", type_=SyftVerifyKey
)
@pytest.fixture(autouse=True)
def verify_key(document_store:DocumentStore) ->PartitionKey:
     return  PartitionKey(
    key="user_verify_key", type_=SyftVerifyKey)