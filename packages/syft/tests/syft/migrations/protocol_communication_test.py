# stdlib
from copy import deepcopy
import os
from pathlib import Path
from unittest import mock

# third party
import pytest

# syft absolute
import syft as sy
from syft.protocol.data_protocol import get_data_protocol
from syft.protocol.data_protocol import protocol_release_dir
from syft.protocol.data_protocol import stage_protocol_changes
from syft.serde.recursive import TYPE_BANK
from syft.serde.serializable import serializable
from syft.server.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.service import AbstractService
from syft.service.service import ServiceConfigRegistry
from syft.service.service import service_method
from syft.service.user.user_roles import GUEST_ROLE_LEVEL
from syft.store.db.db import DBManager
from syft.store.document_store import DocumentStore
from syft.store.document_store import NewBaseStash
from syft.store.document_store import PartitionSettings
from syft.types.syft_migration import migrate
from syft.types.syft_object import SYFT_OBJECT_VERSION_1
from syft.types.syft_object import SyftBaseObject
from syft.types.syft_object import SyftObject
from syft.types.transforms import convert_types
from syft.types.transforms import rename
from syft.types.uid import UID
from syft.util.util import index_syft_by_module_name

MOCK_TYPE_BANK = deepcopy(TYPE_BANK)


def get_klass_version_1():
    @serializable()
    class SyftMockObjectTestV1(SyftObject):
        __canonical_name__ = "SyftMockObjectTest"
        __version__ = SYFT_OBJECT_VERSION_1

        id: UID
        name: str
        version: int

    return SyftMockObjectTestV1


def get_klass_version_2():
    @serializable()
    class SyftMockObjectTestV2(SyftObject):
        __canonical_name__ = "SyftMockObjectTest"
        __version__ = SYFT_OBJECT_VERSION_1

        id: UID
        full_name: str
        version: str

    return SyftMockObjectTestV2


def setup_migration_transforms(mock_klass_v1, mock_klass_v2):
    @migrate(mock_klass_v1, mock_klass_v2)
    def mock_v1_to_v2():
        return [rename("name", "full_name"), convert_types(["version"], str)]

    @migrate(mock_klass_v2, mock_klass_v1)
    def mock_v2_to_v1():
        return [rename("full_name", "name"), convert_types(["version"], int)]

    return mock_v1_to_v2, mock_v2_to_v1


def get_stash_klass(syft_object: type[SyftBaseObject]):
    @serializable(
        canonical_name="SyftMockObjectStash",
        version=1,
    )
    class SyftMockObjectStash(NewBaseStash):
        object_type = syft_object
        settings: PartitionSettings = PartitionSettings(
            name=object_type.__canonical_name__,
            object_type=syft_object,
        )

        def __init__(self, store: DBManager) -> None:
            super().__init__(store=store)

    return SyftMockObjectStash


def setup_service_method(syft_object):
    stash_klass: NewBaseStash = get_stash_klass(syft_object=syft_object)

    @serializable(
        canonical_name="SyftMockObjectService",
        version=1,
    )
    class SyftMockObjectService(AbstractService):
        store: DocumentStore
        stash: stash_klass  # type: ignore
        __module__: str = "syft.test"

        def __init__(self, store: DBManager) -> None:
            self.stash = stash_klass(store=store)

        @service_method(
            path="dummy.syft_object",
            name="get",
            roles=GUEST_ROLE_LEVEL,
        )
        def get(self, context: AuthedServiceContext) -> list[syft_object]:
            return self.stash.get_all(context.credentials, has_permission=True)

    return SyftMockObjectService


def setup_version_one(server_name: str):
    syft_klass_version_one = get_klass_version_1()
    sy.stage_protocol_changes()
    sy.bump_protocol_version()

    syft_service_klass = setup_service_method(
        syft_object=syft_klass_version_one,
    )

    server = sy.orchestra.launch(server_name, dev_mode=True, reset=True)

    worker: Worker = server.python_server

    worker.services.append(syft_service_klass)
    worker.service_path_map[syft_service_klass.__name__.lower()] = syft_service_klass(
        store=worker.document_store
    )

    return server, syft_klass_version_one


def mock_syft_version():
    return f"{sy.__version__}.dev"


def setup_version_second(server_name: str, klass_version_one: type):
    syft_klass_version_second = get_klass_version_2()
    setup_migration_transforms(klass_version_one, syft_klass_version_second)

    sy.stage_protocol_changes()
    sy.bump_protocol_version()

    syft_service_klass = setup_service_method(syft_object=syft_klass_version_second)

    server = sy.orchestra.launch(server_name, dev_mode=True)

    worker: Worker = server.python_server

    worker.services.append(syft_service_klass)
    worker.service_path_map[syft_service_klass.__name__.lower()] = syft_service_klass(
        store=worker.document_store
    )

    return server, syft_klass_version_second


@pytest.fixture
def my_stage_protocol(protocol_file: Path):
    with mock.patch(
        "syft.protocol.data_protocol.PROTOCOL_STATE_FILENAME",
        protocol_file.name,
    ):
        dp = get_data_protocol()
        stage_protocol_changes()
        yield dp.protocol_history
        dp.revert_latest_protocol()
        dp.save_history(dp.protocol_history)

        # Cleanup release dir, remove unused released files
        if os.path.exists(protocol_release_dir()):
            for _file_path in protocol_release_dir().iterdir():
                for version in dp.read_json(_file_path):
                    if version not in dp.protocol_history.keys():
                        _file_path.unlink()


@pytest.mark.skip(
    reason="Issues running with other tests. Shared release folder causes issues."
)
def test_client_server_running_different_protocols(my_stage_protocol):
    def patched_index_syft_by_module_name(fully_qualified_name: str):
        if klass_v1.__name__ in fully_qualified_name:
            return klass_v1
        elif klass_v2.__name__ in fully_qualified_name:
            return klass_v2

        return index_syft_by_module_name(fully_qualified_name)

    server_name = UID().to_string()
    with mock.patch("syft.serde.recursive.TYPE_BANK", MOCK_TYPE_BANK):
        with mock.patch(
            "syft.protocol.data_protocol.TYPE_BANK",
            MOCK_TYPE_BANK,
        ):
            with mock.patch(
                "syft.client.api.index_syft_by_module_name",
                patched_index_syft_by_module_name,
            ):
                # Setup mock object version one
                nh1, klass_v1 = setup_version_one(server_name)
                assert klass_v1.__canonical_name__ == "SyftMockObjectTest"
                assert klass_v1.__name__ == "SyftMockObjectTestV1"

                nh1_client = nh1.client
                assert nh1_client is not None
                result_from_client_1 = nh1_client.api.services.dummy.get()

                protocol_version_with_mock_obj_v1 = get_data_protocol().latest_version

                # No data saved
                assert len(result_from_client_1) == 0

                # Setup mock object version second
                with mock.patch(
                    "syft.protocol.data_protocol.__version__", mock_syft_version()
                ):
                    nh2, klass_v2 = setup_version_second(
                        server_name, klass_version_one=klass_v1
                    )

                    # Create a sample data in version second
                    sample_data = klass_v2(full_name="John", version=str(1), id=UID())

                    assert isinstance(sample_data, klass_v2)

                    # Validate migrations
                    sample_data_v1 = sample_data.migrate_to(
                        version=klass_v1.__version__,
                    )
                    assert sample_data_v1.name == sample_data.full_name
                    assert sample_data_v1.version == int(sample_data.version)

                    # Set the sample data in version second
                    service_klass = nh1.python_server.get_service(
                        "SyftMockObjectService"
                    )
                    service_klass.stash.set(
                        nh1.python_server.root_client.verify_key,
                        sample_data,
                    )

                    nh2_client = nh2.client
                    assert nh2_client is not None
                    # Force communication protocol to when version object is defined
                    nh2_client.communication_protocol = (
                        protocol_version_with_mock_obj_v1
                    )
                    # Reset api
                    nh2_client._api = None

                    # Call the API with an older communication protocol version
                    result2 = nh2_client.api.services.dummy.get()
                    assert isinstance(result2, list)

                    # Validate the data received
                    for data in result2:
                        assert isinstance(data, klass_v1)
                        assert data.name == sample_data.full_name
                        assert data.version == int(sample_data.version)
    ServiceConfigRegistry.__service_config_registry__.pop("dummy.syft_object", None)
