# third party
import pytest

# syft absolute
from syft.__init__ import __version__
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.metadata_service import MetadataService
from syft.core.node.new.metadata_stash import MetadataStash
from syft.core.node.new.node_metadata import NodeMetadata
from syft.core.node.new.node_metadata import NodeMetadataJSON
from syft.core.node.new.node_metadata import NodeMetadataUpdate
from syft.core.node.new.syft_object import HIGHEST_SYFT_OBJECT_VERSION
from syft.core.node.new.syft_object import LOWEST_SYFT_OBJECT_VERSION
from syft.core.node.new.user import User
from syft.core.node.worker import Worker


@pytest.fixture
def metadata_stash(document_store) -> MetadataStash:
    return MetadataStash(store=document_store)


@pytest.fixture
def metadata(worker) -> NodeMetadata:
    return NodeMetadata(
        name=worker.name,
        id=worker.id,
        verify_key=worker.signing_key.verify_key,
        highest_object_version=HIGHEST_SYFT_OBJECT_VERSION,
        lowest_object_version=LOWEST_SYFT_OBJECT_VERSION,
        syft_version=__version__,
    )


@pytest.fixture
def update_metadata(faker) -> NodeMetadataUpdate:
    return NodeMetadataUpdate(
        name=faker.name(),
        description=faker.text(),
        on_board=faker.boolean(),
    )


@pytest.fixture
def metadata_json(faker) -> NodeMetadataJSON:
    return NodeMetadataJSON(
        metadata_version=faker.random_int(),
        name=faker.name(),
        id=faker.text(),
        verify_key=faker.text(),
        highest_object_version=HIGHEST_SYFT_OBJECT_VERSION,
        lowest_object_version=LOWEST_SYFT_OBJECT_VERSION,
        syft_version=__version__,
    )


@pytest.fixture
def metadata_service(document_store) -> MetadataService:
    return MetadataService(store=document_store)


@pytest.fixture
def authed_context(admin_user: User, worker: Worker) -> AuthedServiceContext:
    return AuthedServiceContext(credentials=admin_user.verify_key, node=worker)
