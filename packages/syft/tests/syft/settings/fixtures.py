# stdlib
from datetime import datetime

# third party
import pytest

# syft absolute
from syft.__init__ import __version__
from syft.abstract_server import ServerSideType
from syft.abstract_server import ServerType
from syft.server.credentials import SyftSigningKey
from syft.service.metadata.server_metadata import ServerMetadataJSON
from syft.service.notifier.notifier_stash import NotifierStash
from syft.service.settings.settings import ServerSettings
from syft.service.settings.settings import ServerSettingsUpdate
from syft.service.settings.settings_service import SettingsService
from syft.service.settings.settings_stash import SettingsStash
from syft.types.syft_object import HIGHEST_SYFT_OBJECT_VERSION
from syft.types.syft_object import LOWEST_SYFT_OBJECT_VERSION
from syft.types.uid import UID


@pytest.fixture
def notifier_stash(document_store) -> NotifierStash:
    yield NotifierStash(store=document_store)


@pytest.fixture
def settings_stash(document_store) -> SettingsStash:
    yield SettingsStash(store=document_store)


@pytest.fixture
def settings(worker, faker) -> ServerSettings:
    yield ServerSettings(
        id=UID(),
        name=worker.name,
        organization=faker.text(),
        on_board=faker.boolean(),
        description=faker.text(),
        deployed_on=datetime.now().date().strftime("%m/%d/%Y"),
        signup_enabled=False,
        admin_email="info@openmined.org",
        server_side_type=ServerSideType.LOW_SIDE,
        show_warnings=False,
        verify_key=SyftSigningKey.generate().verify_key,
        server_type=ServerType.DATASITE,
        association_request_auto_approval=False,
        default_worker_pool="default-pool",
        notifications_enabled=False,
    )


@pytest.fixture
def update_settings(faker) -> ServerSettingsUpdate:
    yield ServerSettingsUpdate(
        name=faker.name(),
        description=faker.text(),
        on_board=faker.boolean(),
    )


@pytest.fixture
def metadata_json(faker) -> ServerMetadataJSON:
    yield ServerMetadataJSON(
        metadata_version=faker.random_int(),
        name=faker.name(),
        id=faker.text(),
        verify_key=faker.text(),
        highest_object_version=HIGHEST_SYFT_OBJECT_VERSION,
        lowest_object_version=LOWEST_SYFT_OBJECT_VERSION,
        syft_version=__version__,
        server_side_type=ServerSideType.LOW_SIDE.value,
        show_warnings=False,
        server_type=ServerType.DATASITE.value,
        min_size_blob_storage_mb=1,
    )


@pytest.fixture
def settings_service(document_store) -> SettingsService:
    yield SettingsService(store=document_store)
