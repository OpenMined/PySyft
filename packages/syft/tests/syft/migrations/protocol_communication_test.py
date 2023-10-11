# stdlib
from typing import List
from typing import Type
from typing import Union

# syft absolute
import syft as sy
from syft.node.worker import Worker
from syft.protocol.data_protocol import get_data_protocol
from syft.serde.serializable import serializable
from syft.service.context import AuthedServiceContext
from syft.service.response import SyftError
from syft.service.service import AbstractService
from syft.service.service import service_method
from syft.service.user.user_roles import GUEST_ROLE_LEVEL
from syft.store.document_store import BaseStash
from syft.store.document_store import DocumentStore
from syft.store.document_store import PartitionSettings
from syft.types.syft_migration import migrate
from syft.types.syft_object import SYFT_OBJECT_VERSION_1
from syft.types.syft_object import SYFT_OBJECT_VERSION_2
from syft.types.syft_object import SyftBaseObject
from syft.types.syft_object import SyftObject
from syft.types.transforms import convert_types
from syft.types.transforms import rename
from syft.types.uid import UID
from syft.util.util import set_klass_module_to_syft


def get_klass_version_1():
    @serializable()
    class SyftMockObjectTestV1(SyftObject):
        __canonical_name__ = "SyftMockObjectTest"
        __version__ = SYFT_OBJECT_VERSION_1

        id: UID
        name: str
        version: int
        __module__: str = "syft.test"

    set_klass_module_to_syft(SyftMockObjectTestV1, module_name="test")
    return SyftMockObjectTestV1


def get_klass_version_2():
    @serializable()
    class SyftMockObjectTestV2(SyftObject):
        __canonical_name__ = "SyftMockObjectTest"
        __version__ = SYFT_OBJECT_VERSION_2

        id: UID
        full_name: str
        version: str
        __module__: str = "syft.test"

    set_klass_module_to_syft(SyftMockObjectTestV2, module_name="test")
    return SyftMockObjectTestV2


def setup_migration_transforms(mock_klass_v1, mock_klass_v2):
    @migrate(mock_klass_v1, mock_klass_v2)
    def mock_v1_to_v2():
        return [rename("name", "full_name"), convert_types(["version"], str)]

    @migrate(mock_klass_v2, mock_klass_v1)
    def mock_v2_to_v1():
        return [rename("full_name", "name"), convert_types(["version"], int)]

    return mock_v1_to_v2, mock_v2_to_v1


def get_stash_klass(syft_object: Type[SyftBaseObject]):
    class SyftMockObjectStash(BaseStash):
        object_type = syft_object
        settings: PartitionSettings = PartitionSettings(
            name=object_type.__canonical_name__,
            object_type=syft_object,
        )
        __module__: str = "syft.test"

        def __init__(self, store: DocumentStore) -> None:
            super().__init__(store=store)

    set_klass_module_to_syft(SyftMockObjectStash, module_name="test")
    return SyftMockObjectStash


def setup_service_method(syft_object):
    stash_klass: BaseStash = get_stash_klass(syft_object=syft_object)

    @serializable()
    class SyftMockObjectService(AbstractService):
        store: DocumentStore
        stash: stash_klass
        __module__: str = "syft.test"

        def __init__(self, store: DocumentStore) -> None:
            self.store = store
            self.stash = stash_klass(store=store)

        @service_method(
            path="dummy.syft_object",
            name="get",
            roles=GUEST_ROLE_LEVEL,
        )
        def get(
            self, context: AuthedServiceContext
        ) -> Union[List[syft_object], SyftError]:
            result = self.stash.get_all(context.credentials, has_permission=True)
            if result.is_ok():
                return result.ok()
            return SyftError(message=f"{result.err()}")

    set_klass_module_to_syft(SyftMockObjectService, module_name="test")
    return SyftMockObjectService


def setup_version_one(node_name: str):
    syft_klass_version_one = get_klass_version_1()

    sy.stage_protocol_changes()
    sy.bump_protocol_version()

    syft_service_klass = setup_service_method(
        syft_object=syft_klass_version_one,
    )

    node = sy.orchestra.launch(node_name, dev_mode=True, reset=True)

    worker: Worker = node.python_node

    worker.services.append(syft_service_klass)
    worker.service_path_map[syft_service_klass.__name__.lower()] = syft_service_klass(
        store=worker.document_store
    )

    return node, syft_klass_version_one


def setup_version_second(node_name: str, klass_version_one: type):
    syft_klass_version_second = get_klass_version_2()
    setup_migration_transforms(klass_version_one, syft_klass_version_second)

    sy.stage_protocol_changes()
    sy.bump_protocol_version()

    syft_service_klass = setup_service_method(syft_object=syft_klass_version_second)

    node = sy.orchestra.launch(node_name, dev_mode=True)

    worker: Worker = node.python_node

    worker.services.append(syft_service_klass)
    worker.service_path_map[syft_service_klass.__name__.lower()] = syft_service_klass(
        store=worker.document_store
    )

    return node, syft_klass_version_second


def test_client_server_running_different_protocols():
    node_name = UID().to_string()

    # Setup mock object version one
    nh1, klass_v1 = setup_version_one(node_name)
    assert klass_v1.__canonical_name__ == "SyftMockObjectTest"
    assert klass_v1.__name__ == "SyftMockObjectTestV1"

    nh1_client = nh1.client
    assert nh1_client is not None
    result_from_client_1 = nh1_client.api.services.dummy.get()

    protocol_version_with_mock_obj_v1 = get_data_protocol().latest_version

    # No data saved
    assert len(result_from_client_1) == 0

    # Setup mock object version second
    nh2, klass_v2 = setup_version_second(node_name, klass_version_one=klass_v1)

    # Create a sample data in version second
    sample_data = klass_v2(full_name="John", version=str(1), id=UID())

    assert isinstance(sample_data, klass_v2)

    # Validate migrations
    sample_data_v1 = sample_data.migrate_to(
        version=protocol_version_with_mock_obj_v1,
    )
    assert sample_data_v1.name == sample_data.full_name
    assert sample_data_v1.version == int(sample_data.version)

    # Set the sample data in version second
    service_klass = nh1.python_node.get_service("SyftMockObjectService")
    service_klass.stash.set(
        nh1.python_node.root_client.verify_key,
        sample_data,
    )

    nh2_client = nh2.client
    assert nh2_client is not None
    # Force communication protocol to when version object is defined
    nh2_client.communication_protocol = protocol_version_with_mock_obj_v1
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
