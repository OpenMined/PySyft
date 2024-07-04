# stdlib
from datetime import datetime

# third party
import pytest

# syft absolute
from syft.__init__ import __version__
from syft.abstract_node import NodeSideType
from syft.abstract_node import NodeType
from syft.node.credentials import SyftSigningKey
from syft.service.metadata.node_metadata import NodeMetadataJSON
from syft.service.settings.settings import NodeSettings
from syft.service.settings.settings import NodeSettingsUpdate
from syft.service.settings.settings_service import SettingsService
from syft.service.settings.settings_stash import SettingsStash
from syft.types.syft_object import HIGHEST_SYFT_OBJECT_VERSION
from syft.types.syft_object import LOWEST_SYFT_OBJECT_VERSION
from syft.types.uid import UID


@pytest.fixture()
def settings_stash(document_store) -> SettingsStash:
    return SettingsStash(store=document_store)


@pytest.fixture()
def settings(worker, faker) -> NodeSettings:
    return NodeSettings(
        id=UID(),
        name=worker.name,
        organization=faker.text(),
        on_board=faker.boolean(),
        description=faker.text(),
        deployed_on=datetime.now().date().strftime("%m/%d/%Y"),
        signup_enabled=False,
        admin_email="info@openmined.org",
        node_side_type=NodeSideType.LOW_SIDE,
        show_warnings=False,
        verify_key=SyftSigningKey.generate().verify_key,
        node_type=NodeType.DOMAIN,
        association_request_auto_approval=False,
        default_worker_pool="default-pool",
    )


@pytest.fixture()
def update_settings(faker) -> NodeSettingsUpdate:
    return NodeSettingsUpdate(
        name=faker.name(),
        description=faker.text(),
        on_board=faker.boolean(),
    )


@pytest.fixture()
def metadata_json(faker) -> NodeMetadataJSON:
    return NodeMetadataJSON(
        metadata_version=faker.random_int(),
        name=faker.name(),
        id=faker.text(),
        verify_key=faker.text(),
        highest_object_version=HIGHEST_SYFT_OBJECT_VERSION,
        lowest_object_version=LOWEST_SYFT_OBJECT_VERSION,
        syft_version=__version__,
        node_side_type=NodeSideType.LOW_SIDE.value,
        show_warnings=False,
        node_type=NodeType.DOMAIN.value,
        min_size_blob_storage_mb=16,
    )


@pytest.fixture()
def settings_service(document_store) -> SettingsService:
    return SettingsService(store=document_store)
