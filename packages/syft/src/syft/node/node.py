# future
from __future__ import annotations

# stdlib
import binascii
from collections import OrderedDict
import contextlib
from datetime import datetime
from functools import partial
import hashlib
from multiprocessing import current_process
import os
from pathlib import Path
import subprocess  # nosec
import traceback
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
import uuid

# third party
from nacl.signing import SigningKey
from result import Err
from result import Result
from typing_extensions import Self

# relative
from .. import __version__
from ..abstract_node import AbstractNode
from ..abstract_node import NodeSideType
from ..abstract_node import NodeType
from ..client.api import SignedSyftAPICall
from ..client.api import SyftAPI
from ..client.api import SyftAPICall
from ..client.api import SyftAPIData
from ..client.api import debox_signed_syftapicall_response
from ..exceptions.exception import PySyftException
from ..external import OBLV
from ..protocol.data_protocol import PROTOCOL_TYPE
from ..protocol.data_protocol import get_data_protocol
from ..service.action.action_object import Action
from ..service.action.action_object import ActionObject
from ..service.action.action_service import ActionService
from ..service.action.action_store import DictActionStore
from ..service.action.action_store import MongoActionStore
from ..service.action.action_store import SQLiteActionStore
from ..service.blob_storage.service import BlobStorageService
from ..service.code.user_code_service import UserCodeService
from ..service.code.user_code_stash import UserCodeStash
from ..service.code_history.code_history_service import CodeHistoryService
from ..service.context import AuthedServiceContext
from ..service.context import NodeServiceContext
from ..service.context import UnauthedServiceContext
from ..service.context import UserLoginCredentials
from ..service.data_subject.data_subject_member_service import DataSubjectMemberService
from ..service.data_subject.data_subject_service import DataSubjectService
from ..service.dataset.dataset_service import DatasetService
from ..service.enclave.enclave_service import EnclaveService
from ..service.job.job_service import JobService
from ..service.job.job_stash import Job
from ..service.log.log_service import LogService
from ..service.metadata.metadata_service import MetadataService
from ..service.metadata.node_metadata import NodeMetadataV3
from ..service.network.network_service import NetworkService
from ..service.notification.notification_service import NotificationService
from ..service.object_search.migration_state_service import MigrateStateService
from ..service.policy.policy_service import PolicyService
from ..service.project.project_service import ProjectService
from ..service.queue.base_queue import AbstractMessageHandler
from ..service.queue.base_queue import QueueConsumer
from ..service.queue.base_queue import QueueProducer
from ..service.queue.queue import APICallMessageHandler
from ..service.queue.queue import QueueManager
from ..service.queue.queue_service import QueueService
from ..service.queue.queue_stash import ActionQueueItem
from ..service.queue.queue_stash import QueueItem
from ..service.queue.queue_stash import QueueStash
from ..service.queue.zmq_queue import QueueConfig
from ..service.queue.zmq_queue import ZMQClientConfig
from ..service.queue.zmq_queue import ZMQQueueConfig
from ..service.request.request_service import RequestService
from ..service.response import SyftError
from ..service.service import AbstractService
from ..service.service import ServiceConfigRegistry
from ..service.service import UserServiceConfigRegistry
from ..service.settings.settings import NodeSettingsV2
from ..service.settings.settings_service import SettingsService
from ..service.settings.settings_stash import SettingsStash
from ..service.user.user import User
from ..service.user.user import UserCreate
from ..service.user.user_roles import ServiceRole
from ..service.user.user_service import UserService
from ..service.user.user_stash import UserStash
from ..service.worker.image_registry_service import SyftImageRegistryService
from ..service.worker.utils import DEFAULT_WORKER_IMAGE_TAG
from ..service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ..service.worker.utils import create_default_image
from ..service.worker.worker_image_service import SyftWorkerImageService
from ..service.worker.worker_pool_service import SyftWorkerPoolService
from ..service.worker.worker_pool_stash import SyftWorkerPoolStash
from ..service.worker.worker_service import WorkerService
from ..store.blob_storage import BlobStorageConfig
from ..store.blob_storage.on_disk import OnDiskBlobStorageClientConfig
from ..store.blob_storage.on_disk import OnDiskBlobStorageConfig
from ..store.dict_document_store import DictStoreConfig
from ..store.document_store import StoreConfig
from ..store.linked_obj import LinkedObject
from ..store.mongo_document_store import MongoStoreConfig
from ..store.sqlite_document_store import SQLiteStoreClientConfig
from ..store.sqlite_document_store import SQLiteStoreConfig
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID
from ..util.experimental_flags import flags
from ..util.telemetry import instrument
from ..util.util import get_env
from ..util.util import get_queue_address
from ..util.util import get_root_data_path
from ..util.util import random_name
from ..util.util import str_to_bool
from ..util.util import thread_ident
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .worker_settings import WorkerSettings

# if user code needs to be serded and its not available we can call this to refresh
# the code for a specific node UID and thread
CODE_RELOADER: Dict[int, Callable] = {}


NODE_PRIVATE_KEY = "NODE_PRIVATE_KEY"
NODE_UID = "NODE_UID"
NODE_TYPE = "NODE_TYPE"
NODE_NAME = "NODE_NAME"
NODE_SIDE_TYPE = "NODE_SIDE_TYPE"

DEFAULT_ROOT_EMAIL = "DEFAULT_ROOT_EMAIL"
DEFAULT_ROOT_USERNAME = "DEFAULT_ROOT_USERNAME"
DEFAULT_ROOT_PASSWORD = "DEFAULT_ROOT_PASSWORD"  # nosec


def get_private_key_env() -> Optional[str]:
    return get_env(NODE_PRIVATE_KEY)


def get_node_type() -> Optional[str]:
    return get_env(NODE_TYPE, "domain")


def get_node_name() -> Optional[str]:
    return get_env(NODE_NAME, None)


def get_node_side_type() -> str:
    return get_env(NODE_SIDE_TYPE, "high")


def get_node_uid_env() -> Optional[str]:
    return get_env(NODE_UID)


def get_default_root_email() -> Optional[str]:
    return get_env(DEFAULT_ROOT_EMAIL, "info@openmined.org")


def get_default_root_username() -> Optional[str]:
    return get_env(DEFAULT_ROOT_USERNAME, "Jane Doe")


def get_default_root_password() -> Optional[str]:
    return get_env(DEFAULT_ROOT_PASSWORD, "changethis")  # nosec


def get_dev_mode() -> bool:
    return str_to_bool(get_env("DEV_MODE", "False"))


def get_enable_warnings() -> bool:
    return str_to_bool(get_env("ENABLE_WARNINGS", "False"))


def get_container_host() -> Optional[str]:
    return get_env("CONTAINER_HOST")


def get_default_worker_image() -> str:
    return get_env("DEFAULT_WORKER_POOL_IMAGE")


def get_default_worker_pool_name() -> str:
    return get_env("DEFAULT_WORKER_POOL_NAME", DEFAULT_WORKER_POOL_NAME)


def get_default_worker_pool_count() -> int:
    return int(get_env("DEFAULT_WORKER_POOL_COUNT", 1))


def in_kubernetes() -> Optional[str]:
    return get_container_host() == "k8s"


def get_venv_packages() -> str:
    res = subprocess.getoutput(
        "pip list --format=freeze",
    )
    return res


def get_syft_worker() -> bool:
    return str_to_bool(get_env("SYFT_WORKER", "false"))


def get_k8s_pod_name() -> Optional[str]:
    return get_env("K8S_POD_NAME")


def get_syft_worker_uid() -> Optional[str]:
    is_worker = get_syft_worker()
    pod_name = get_k8s_pod_name()
    uid = get_env("SYFT_WORKER_UID")
    # if uid is empty is a K8S worker, generate a uid from the pod name
    if (not uid) and is_worker and pod_name:
        uid = str(UID.with_seed(pod_name))
    return uid


signing_key_env = get_private_key_env()
node_uid_env = get_node_uid_env()

default_root_email = get_default_root_email()
default_root_username = get_default_root_username()
default_root_password = get_default_root_password()


class AuthNodeContextRegistry:
    __node_context_registry__: Dict[Tuple, Node] = OrderedDict()

    @classmethod
    def set_node_context(
        cls,
        node_uid: Union[UID, str],
        context: NodeServiceContext,
        user_verify_key: Union[SyftVerifyKey, str],
    ):
        if isinstance(node_uid, str):
            node_uid = UID.from_string(node_uid)

        if isinstance(user_verify_key, str):
            user_verify_key = SyftVerifyKey.from_string(user_verify_key)

        key = cls._get_key(node_uid=node_uid, user_verify_key=user_verify_key)

        cls.__node_context_registry__[key] = context

    @staticmethod
    def _get_key(node_uid: UID, user_verify_key: SyftVerifyKey) -> str:
        return "-".join(str(x) for x in (node_uid, user_verify_key))

    @classmethod
    def auth_context_for_user(
        cls,
        node_uid: UID,
        user_verify_key: SyftVerifyKey,
    ) -> Optional[AuthedServiceContext]:
        key = cls._get_key(node_uid=node_uid, user_verify_key=user_verify_key)
        return cls.__node_context_registry__.get(key)


@instrument
class Node(AbstractNode):
    signing_key: Optional[SyftSigningKey]
    required_signed_calls: bool = True
    packages: str

    def __init__(
        self,
        *,  # Trasterisk
        name: Optional[str] = None,
        id: Optional[UID] = None,
        services: Optional[List[Type[AbstractService]]] = None,
        signing_key: Optional[Union[SyftSigningKey, SigningKey]] = None,
        action_store_config: Optional[StoreConfig] = None,
        document_store_config: Optional[StoreConfig] = None,
        root_email: str = default_root_email,
        root_username: str = default_root_username,
        root_password: str = default_root_password,
        processes: int = 0,
        is_subprocess: bool = False,
        node_type: Union[str, NodeType] = NodeType.DOMAIN,
        local_db: bool = False,
        sqlite_path: Optional[str] = None,
        blob_storage_config: Optional[BlobStorageConfig] = None,
        queue_config: Optional[QueueConfig] = None,
        queue_port: Optional[int] = None,
        n_consumers: int = 0,
        create_producer: bool = False,
        thread_workers: bool = False,
        node_side_type: Union[str, NodeSideType] = NodeSideType.HIGH_SIDE,
        enable_warnings: bool = False,
        dev_mode: bool = False,
        migrate: bool = False,
        in_memory_workers: bool = True,
    ):
        # 🟡 TODO 22: change our ENV variable format and default init args to make this
        # less horrible or add some convenience functions
        self.dev_mode = dev_mode or get_dev_mode()
        if node_uid_env is not None:
            self.id = UID.from_string(node_uid_env)
        else:
            if id is None:
                id = UID()
            self.id = id
        self.packages = ""

        self.signing_key = None
        if signing_key_env is not None:
            self.signing_key = SyftSigningKey.from_string(signing_key_env)
        else:
            if isinstance(signing_key, SigningKey):
                signing_key = SyftSigningKey(signing_key=signing_key)
            self.signing_key = signing_key

        if self.signing_key is None:
            self.signing_key = SyftSigningKey.generate()

        self.processes = processes
        self.is_subprocess = is_subprocess
        self.name = random_name() if name is None else name
        services = (
            [
                UserService,
                WorkerService,
                SettingsService,
                ActionService,
                LogService,
                DatasetService,
                UserCodeService,
                QueueService,
                JobService,
                RequestService,
                DataSubjectService,
                NetworkService,
                PolicyService,
                NotificationService,
                DataSubjectMemberService,
                ProjectService,
                EnclaveService,
                CodeHistoryService,
                MetadataService,
                BlobStorageService,
                MigrateStateService,
                SyftWorkerImageService,
                SyftWorkerPoolService,
                SyftImageRegistryService,
            ]
            if services is None
            else services
        )

        self.service_config = ServiceConfigRegistry.get_registered_configs()
        self.local_db = local_db
        self.sqlite_path = sqlite_path
        self.init_stores(
            action_store_config=action_store_config,
            document_store_config=document_store_config,
        )

        if OBLV:
            # relative
            from ..external.oblv.oblv_service import OblvService

            services += [OblvService]
            create_oblv_key_pair(worker=self)

        self.enable_warnings = enable_warnings
        self.in_memory_workers = in_memory_workers

        self.services = services
        self._construct_services()

        create_admin_new(  # nosec B106
            name=root_username,
            email=root_email,
            password=root_password,
            node=self,
        )

        self.client_cache = {}
        if isinstance(node_type, str):
            node_type = NodeType(node_type)
        self.node_type = node_type

        if isinstance(node_side_type, str):
            node_side_type = NodeSideType(node_side_type)
        self.node_side_type = node_side_type

        self.post_init()
        self.create_initial_settings(admin_email=root_email)

        self.queue_config = self.create_queue_config(
            n_consumers=n_consumers,
            create_producer=create_producer,
            thread_workers=thread_workers,
            queue_port=queue_port,
            queue_config=queue_config,
        )

        self.init_queue_manager(queue_config=self.queue_config)

        self.init_blob_storage(config=blob_storage_config)

        # Migrate data before any operation on db
        if migrate:
            self.find_and_migrate_data()

        NodeRegistry.set_node_for(self.id, self)

    @property
    def runs_in_docker(self):
        path = "/proc/self/cgroup"
        return (
            os.path.exists("/.dockerenv")
            or os.path.isfile(path)
            and any("docker" in line for line in open(path))
        )

    def init_blob_storage(self, config: Optional[BlobStorageConfig] = None) -> None:
        if config is None:
            root_directory = get_root_data_path()
            base_directory = root_directory / f"{self.id}"
            client_config = OnDiskBlobStorageClientConfig(base_directory=base_directory)
            config_ = OnDiskBlobStorageConfig(client_config=client_config)
        else:
            config_ = config
        self.blob_store_config = config_
        self.blob_storage_client = config_.client_type(config=config_.client_config)

        # relative
        from ..store.blob_storage.seaweedfs import SeaweedFSConfig

        if isinstance(config, SeaweedFSConfig):
            blob_storage_service = self.get_service(BlobStorageService)
            remote_profiles = blob_storage_service.remote_profile_stash.get_all(
                credentials=self.signing_key.verify_key, has_permission=True
            ).ok()
            for remote_profile in remote_profiles:
                self.blob_store_config.client_config.remote_profiles[
                    remote_profile.profile_name
                ] = remote_profile

    def stop(self):
        for consumer_list in self.queue_manager.consumers.values():
            for c in consumer_list:
                c.close()
        for p in self.queue_manager.producers.values():
            p.close()

    def close(self):
        self.stop()

    def create_queue_config(
        self,
        n_consumers: int,
        create_producer: bool,
        thread_workers: bool,
        queue_port: Optional[int],
        queue_config: Optional[QueueConfig],
    ) -> QueueConfig:
        if queue_config:
            queue_config_ = queue_config
        elif queue_port is not None or n_consumers > 0 or create_producer:
            if not create_producer and queue_port is None:
                print("No queue port defined to bind consumers.")
            queue_config_ = ZMQQueueConfig(
                client_config=ZMQClientConfig(
                    create_producer=create_producer,
                    queue_port=queue_port,
                    n_consumers=n_consumers,
                ),
                thread_workers=thread_workers,
            )
        else:
            queue_config_ = ZMQQueueConfig()

        return queue_config_

    def init_queue_manager(self, queue_config: QueueConfig):
        MessageHandlers = [APICallMessageHandler]
        if self.is_subprocess:
            return

        self.queue_manager = QueueManager(config=queue_config)
        for message_handler in MessageHandlers:
            queue_name = message_handler.queue_name
            # client config
            if getattr(queue_config.client_config, "create_producer", True):
                context = AuthedServiceContext(
                    node=self,
                    credentials=self.verify_key,
                    role=ServiceRole.ADMIN,
                )
                producer: QueueProducer = self.queue_manager.create_producer(
                    queue_name=queue_name,
                    queue_stash=self.queue_stash,
                    context=context,
                    worker_stash=self.worker_stash,
                )
                producer.run()
                address = producer.address
            else:
                port = queue_config.client_config.queue_port
                if port is not None:
                    address = get_queue_address(port)
                else:
                    address = None

            if address is None and queue_config.client_config.n_consumers > 0:
                raise ValueError("address unknown for consumers")

            service_name = queue_config.client_config.consumer_service

            if not service_name:
                # Create consumers for default worker pool
                create_default_worker_pool(self)
            else:
                # Create consumer for given worker pool
                syft_worker_uid = get_syft_worker_uid()
                print(
                    f"Running as consumer with uid={syft_worker_uid} service={service_name}"
                )

                if syft_worker_uid:
                    self.add_consumer_for_service(
                        service_name=service_name,
                        syft_worker_id=UID(syft_worker_uid),
                        address=address,
                        message_handler=message_handler,
                    )

    def add_consumer_for_service(
        self,
        service_name: str,
        syft_worker_id: UID,
        address: str,
        message_handler: AbstractMessageHandler = APICallMessageHandler,
    ):
        consumer: QueueConsumer = self.queue_manager.create_consumer(
            message_handler,
            address=address,
            service_name=service_name,
            worker_stash=self.worker_stash,
            syft_worker_id=syft_worker_id,
        )
        consumer.run()

    @classmethod
    def named(
        cls,
        *,  # Trasterisk
        name: str,
        processes: int = 0,
        reset: bool = False,
        local_db: bool = False,
        sqlite_path: Optional[str] = None,
        node_type: Union[str, NodeType] = NodeType.DOMAIN,
        node_side_type: Union[str, NodeSideType] = NodeSideType.HIGH_SIDE,
        enable_warnings: bool = False,
        n_consumers: int = 0,
        thread_workers: bool = False,
        create_producer: bool = False,
        queue_port: Optional[int] = None,
        dev_mode: bool = False,
        migrate: bool = False,
        in_memory_workers: bool = True,
    ) -> Self:
        name_hash = hashlib.sha256(name.encode("utf8")).digest()
        name_hash_uuid = name_hash[0:16]
        name_hash_uuid = bytearray(name_hash_uuid)
        name_hash_uuid[6] = (
            name_hash_uuid[6] & 0x0F
        ) | 0x40  # Set version to 4 (uuid4)
        name_hash_uuid[8] = (name_hash_uuid[8] & 0x3F) | 0x80  # Set variant to RFC 4122
        name_hash_string = binascii.hexlify(bytearray(name_hash_uuid)).decode("utf-8")
        if uuid.UUID(name_hash_string).version != 4:
            raise Exception(f"Invalid UID: {name_hash_string} for name: {name}")
        uid = UID(name_hash_string)
        key = SyftSigningKey(signing_key=SigningKey(name_hash))
        blob_storage_config = None
        if reset:
            store_config = SQLiteStoreClientConfig()
            store_config.filename = f"{uid}.sqlite"

            # stdlib
            import sqlite3

            with contextlib.closing(sqlite3.connect(store_config.file_path)) as db:
                cursor = db.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                for table_name in tables:
                    drop_table_sql = f"DROP TABLE IF EXISTS {table_name[0]};"
                    cursor.execute(drop_table_sql)

                db.commit()
                db.close()

            # remove lock files for reading
            # we should update this to partition locks per node
            for f in Path("/tmp/sherlock").glob("*.json"):  # nosec
                if f.is_file():
                    f.unlink()

            with contextlib.suppress(FileNotFoundError, PermissionError):
                if os.path.exists(store_config.file_path):
                    os.unlink(store_config.file_path)

            # Reset blob storage
            root_directory = get_root_data_path()
            base_directory = root_directory / f"{uid}"
            if base_directory.exists():
                for file in base_directory.iterdir():
                    file.unlink()
            blob_client_config = OnDiskBlobStorageClientConfig(
                base_directory=base_directory
            )
            blob_storage_config = OnDiskBlobStorageConfig(
                client_config=blob_client_config
            )

        node_type = NodeType(node_type)
        node_side_type = NodeSideType(node_side_type)

        return cls(
            name=name,
            id=uid,
            signing_key=key,
            processes=processes,
            local_db=local_db,
            sqlite_path=sqlite_path,
            node_type=node_type,
            node_side_type=node_side_type,
            enable_warnings=enable_warnings,
            blob_storage_config=blob_storage_config,
            queue_port=queue_port,
            n_consumers=n_consumers,
            thread_workers=thread_workers,
            create_producer=create_producer,
            dev_mode=dev_mode,
            migrate=migrate,
            in_memory_workers=in_memory_workers,
        )

    def is_root(self, credentials: SyftVerifyKey) -> bool:
        return credentials == self.verify_key

    @property
    def root_client(self):
        # relative
        from ..client.client import PythonConnection

        connection = PythonConnection(node=self)
        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type
        root_client = client_type(connection=connection, credentials=self.signing_key)
        root_client.api.refresh_api_callback()
        return root_client

    def _find_klasses_pending_for_migration(
        self, object_types: List[SyftObject]
    ) -> List[SyftObject]:
        context = AuthedServiceContext(
            node=self,
            credentials=self.verify_key,
            role=ServiceRole.ADMIN,
        )
        migration_state_service = self.get_service(MigrateStateService)

        klasses_to_be_migrated = []

        for object_type in object_types:
            canonical_name = object_type.__canonical_name__
            object_version = object_type.__version__

            migration_state = migration_state_service.get_state(context, canonical_name)
            if (
                migration_state is not None
                and migration_state.current_version != migration_state.latest_version
            ):
                klasses_to_be_migrated.append(object_type)
            else:
                migration_state_service.register_migration_state(
                    context,
                    current_version=object_version,
                    canonical_name=canonical_name,
                )

        return klasses_to_be_migrated

    def find_and_migrate_data(self):
        # Track all object type that need migration for document store
        context = AuthedServiceContext(
            node=self,
            credentials=self.verify_key,
            role=ServiceRole.ADMIN,
        )
        document_store_object_types = [
            partition.settings.object_type
            for partition in self.document_store.partitions.values()
        ]

        object_pending_migration = self._find_klasses_pending_for_migration(
            object_types=document_store_object_types
        )

        if object_pending_migration:
            print(
                "Object in Document Store that needs migration: ",
                object_pending_migration,
            )

        # Migrate data for objects in document store
        for object_type in object_pending_migration:
            canonical_name = object_type.__canonical_name__
            object_partition = self.document_store.partitions.get(canonical_name)
            if object_partition is None:
                continue

            print(f"Migrating data for: {canonical_name} table.")
            migration_status = object_partition.migrate_data(
                to_klass=object_type, context=context
            )
            if migration_status.is_err():
                raise Exception(
                    f"Failed to migrate data for {canonical_name}. Error: {migration_status.err()}"
                )

        # Track all object types from action store
        action_object_types = [Action, ActionObject]
        action_object_types.extend(ActionObject.__subclasses__())
        action_object_pending_migration = self._find_klasses_pending_for_migration(
            action_object_types
        )

        if action_object_pending_migration:
            print(
                "Object in Action Store that needs migration: ",
                action_object_pending_migration,
            )

        # Migrate data for objects in action store
        for object_type in action_object_pending_migration:
            canonical_name = object_type.__canonical_name__

            migration_status = self.action_store.migrate_data(
                to_klass=object_type, credentials=self.verify_key
            )
            if migration_status.is_err():
                raise Exception(
                    f"Failed to migrate data for {canonical_name}. Error: {migration_status.err()}"
                )
        print("Data Migrated to latest version !!!")

    @property
    def guest_client(self):
        return self.get_guest_client()

    @property
    def current_protocol(self) -> List:
        data_protocol = get_data_protocol()
        return data_protocol.latest_version

    def get_guest_client(self, verbose: bool = True):
        # relative
        from ..client.client import PythonConnection

        connection = PythonConnection(node=self)
        if verbose:
            print(
                f"Logged into <{self.name}: {self.node_side_type.value.capitalize()} "
                f"side {self.node_type.value.capitalize()} > as GUEST"
            )

        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type

        guest_client = client_type(
            connection=connection, credentials=SyftSigningKey.generate()
        )
        guest_client.api.refresh_api_callback()
        return guest_client

    def __repr__(self) -> str:
        service_string = ""
        if not self.is_subprocess:
            services = []
            for service in self.services:
                services.append(service.__name__)
            service_string = ", ".join(sorted(services))
            service_string = f"\n\nServices:\n{service_string}"
        return f"{type(self).__name__}: {self.name} - {self.id} - {self.node_type}{service_string}"

    def post_init(self) -> None:
        context = AuthedServiceContext(
            node=self, credentials=self.verify_key, role=ServiceRole.ADMIN
        )
        AuthNodeContextRegistry.set_node_context(
            node_uid=self.id, user_verify_key=self.verify_key, context=context
        )

        if UserCodeService in self.services:
            user_code_service = self.get_service(UserCodeService)
            user_code_service.load_user_code(context=context)

        if self.is_subprocess or current_process().name != "MainProcess":
            # print(f"> Starting Subprocess {self}")
            pass
        else:
            pass
            # why would we do this?
            # print(f"> {self}")

        def reload_user_code() -> None:
            user_code_service.load_user_code(context=context)

        CODE_RELOADER[thread_ident()] = reload_user_code

    def init_stores(
        self,
        document_store_config: Optional[StoreConfig] = None,
        action_store_config: Optional[StoreConfig] = None,
    ):
        if document_store_config is None:
            if self.local_db or (self.processes > 0 and not self.is_subprocess):
                client_config = SQLiteStoreClientConfig(path=self.sqlite_path)
                document_store_config = SQLiteStoreConfig(client_config=client_config)
            else:
                document_store_config = DictStoreConfig()
        if (
            isinstance(document_store_config, SQLiteStoreConfig)
            and document_store_config.client_config.filename is None
        ):
            document_store_config.client_config.filename = f"{self.id}.sqlite"
            if self.dev_mode:
                print(
                    f"SQLite Store Path:\n!open file://{document_store_config.client_config.file_path}\n"
                )
        document_store = document_store_config.store_type
        self.document_store_config = document_store_config

        # We add the python id of the current node in order
        # to create one connection per Node object in MongoClientCache
        # so that we avoid closing the connection from a
        # different thread through the garbage collection
        if isinstance(self.document_store_config, MongoStoreConfig):
            self.document_store_config.client_config.node_obj_python_id = id(self)

        self.document_store = document_store(
            root_verify_key=self.verify_key,
            store_config=document_store_config,
        )
        if action_store_config is None:
            if self.local_db or (self.processes > 0 and not self.is_subprocess):
                client_config = SQLiteStoreClientConfig(path=self.sqlite_path)
                action_store_config = SQLiteStoreConfig(client_config=client_config)
            else:
                action_store_config = DictStoreConfig()

        if (
            isinstance(action_store_config, SQLiteStoreConfig)
            and action_store_config.client_config.filename is None
        ):
            action_store_config.client_config.filename = f"{self.id}.sqlite"

        if isinstance(action_store_config, SQLiteStoreConfig):
            self.action_store = SQLiteActionStore(
                store_config=action_store_config,
                root_verify_key=self.verify_key,
            )
        elif isinstance(action_store_config, MongoStoreConfig):
            # We add the python id of the current node in order
            # to create one connection per Node object in MongoClientCache
            # so that we avoid closing the connection from a
            # different thread through the garbage collection
            action_store_config.client_config.node_obj_python_id = id(self)

            self.action_store = MongoActionStore(
                root_verify_key=self.verify_key, store_config=action_store_config
            )
        else:
            self.action_store = DictActionStore(root_verify_key=self.verify_key)

        self.action_store_config = action_store_config
        self.queue_stash = QueueStash(store=self.document_store)

    @property
    def job_stash(self):
        return self.get_service("jobservice").stash

    @property
    def worker_stash(self):
        return self.get_service("workerservice").stash

    def _construct_services(self):
        self.service_path_map = {}

        for service_klass in self.services:
            kwargs = {}
            if service_klass == ActionService:
                kwargs["store"] = self.action_store
            store_services = [
                UserService,
                WorkerService,
                SettingsService,
                DatasetService,
                UserCodeService,
                LogService,
                RequestService,
                QueueService,
                JobService,
                DataSubjectService,
                NetworkService,
                PolicyService,
                NotificationService,
                DataSubjectMemberService,
                ProjectService,
                EnclaveService,
                CodeHistoryService,
                MetadataService,
                BlobStorageService,
                MigrateStateService,
                SyftWorkerImageService,
                SyftWorkerPoolService,
                SyftImageRegistryService,
            ]

            if OBLV:
                # relative
                from ..external.oblv.oblv_service import OblvService

                store_services += [OblvService]

            if service_klass in store_services:
                kwargs["store"] = self.document_store
            self.service_path_map[service_klass.__name__.lower()] = service_klass(
                **kwargs
            )

    def get_service_method(self, path_or_func: Union[str, Callable]) -> Callable:
        if callable(path_or_func):
            path_or_func = path_or_func.__qualname__
        return self._get_service_method_from_path(path_or_func)

    def get_service(self, path_or_func: Union[str, Callable]) -> Callable:
        if callable(path_or_func):
            path_or_func = path_or_func.__qualname__
        return self._get_service_from_path(path_or_func)

    def _get_service_from_path(self, path: str) -> AbstractService:
        path_list = path.split(".")
        if len(path_list) > 1:
            _ = path_list.pop()
        service_name = path_list.pop()
        return self.service_path_map[service_name.lower()]

    def _get_service_method_from_path(self, path: str) -> Callable:
        path_list = path.split(".")
        method_name = path_list.pop()
        service_obj = self._get_service_from_path(path=path)

        return getattr(service_obj, method_name)

    @property
    def settings(self) -> NodeSettingsV2:
        settings_stash = SettingsStash(store=self.document_store)
        settings = settings_stash.get_all(self.signing_key.verify_key)
        if settings.is_ok() and len(settings.ok()) > 0:
            settings_data = settings.ok()[0]
        return settings_data

    @property
    def metadata(self) -> NodeMetadataV3:
        name = ""
        organization = ""
        description = ""
        show_warnings = self.enable_warnings
        settings_data = self.settings
        name = settings_data.name
        organization = settings_data.organization
        description = settings_data.description
        show_warnings = settings_data.show_warnings

        return NodeMetadataV3(
            name=name,
            id=self.id,
            verify_key=self.verify_key,
            highest_version=SYFT_OBJECT_VERSION_1,
            lowest_version=SYFT_OBJECT_VERSION_1,
            syft_version=__version__,
            description=description,
            organization=organization,
            node_type=self.node_type.value,
            node_side_type=self.node_side_type.value,
            show_warnings=show_warnings,
        )

    @property
    def icon(self) -> str:
        return "🦾"

    @property
    def verify_key(self) -> SyftVerifyKey:
        return self.signing_key.verify_key

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False

        if self.id != other.id:
            return False

        return True

    def await_future(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Union[Optional[QueueItem], SyftError]:
        # stdlib
        from time import sleep

        # relative
        from ..service.queue.queue import Status

        while True:
            result = self.queue_stash.pop_on_complete(credentials, uid)
            if not result.is_ok():
                return result.err()
            else:
                res = result.ok()
                if res.status == Status.COMPLETED:
                    return res
            sleep(0.1)

    def resolve_future(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Union[Optional[QueueItem], SyftError]:
        result = self.queue_stash.pop_on_complete(credentials, uid)

        if result.is_ok():
            queue_obj = result.ok()
            queue_obj._set_obj_location_(
                node_uid=self.id,
                credentials=credentials,
            )
            return queue_obj
        return result.err()

    def forward_message(
        self, api_call: Union[SyftAPICall, SignedSyftAPICall]
    ) -> Result[Union[QueueItem, SyftObject], Err]:
        node_uid = api_call.message.node_uid
        if NetworkService not in self.services:
            return SyftError(
                message=(
                    "Node has no network service so we can't "
                    f"forward this message to {node_uid}"
                )
            )

        client = None
        if node_uid in self.client_cache:
            client = self.client_cache[node_uid]
        else:
            network_service = self.get_service(NetworkService)
            peer = network_service.stash.get_by_uid(self.verify_key, node_uid)

            if peer.is_ok() and peer.ok():
                peer = peer.ok()
                context = AuthedServiceContext(
                    node=self, credentials=api_call.credentials
                )
                client = peer.client_with_context(context=context)
                self.client_cache[node_uid] = client
        if client:
            message: SyftAPICall = api_call.message
            if message.path == "metadata":
                result = client.metadata
            elif message.path == "login":
                result = client.connection.login(**message.kwargs)
            elif message.path == "register":
                result = client.connection.register(**message.kwargs)
            elif message.path == "api":
                result = client.connection.get_api(**message.kwargs)
            else:
                signed_result = client.connection.make_call(api_call)
                result = debox_signed_syftapicall_response(signed_result=signed_result)

                # relative
                from ..store.blob_storage import BlobRetrievalByURL

                # In the case of blob storage, the gateway downloads the result and then passes it to
                # the proxy client
                if isinstance(result, BlobRetrievalByURL):
                    blob_route = client.api.connection.to_blob_route(
                        result.url.url_path
                    )
                    result.url = blob_route
                    final_res = result.read()
                    return final_res

            return result

        return SyftError(message=(f"Node has no route to {node_uid}"))

    def get_role_for_credentials(self, credentials: SyftVerifyKey) -> ServiceRole:
        role = self.get_service("userservice").get_role_for_credentials(
            credentials=credentials
        )
        return role

    def handle_api_call(
        self,
        api_call: Union[SyftAPICall, SignedSyftAPICall],
        job_id: Optional[UID] = None,
        check_call_location=True,
    ) -> Result[SignedSyftAPICall, Err]:
        # Get the result
        result = self.handle_api_call_with_unsigned_result(
            api_call, job_id=job_id, check_call_location=check_call_location
        )
        # Sign the result
        signed_result = SyftAPIData(data=result).sign(self.signing_key)

        return signed_result

    def handle_api_call_with_unsigned_result(
        self,
        api_call: Union[SyftAPICall, SignedSyftAPICall],
        job_id: Optional[UID] = None,
        check_call_location=True,
    ) -> Result[Union[QueueItem, SyftObject], Err]:
        if self.required_signed_calls and isinstance(api_call, SyftAPICall):
            return SyftError(
                message=f"You sent a {type(api_call)}. This node requires SignedSyftAPICall."  # type: ignore
            )
        else:
            if not api_call.is_valid:
                return SyftError(message="Your message signature is invalid")  # type: ignore

        if api_call.message.node_uid != self.id and check_call_location:
            return self.forward_message(api_call=api_call)
        if api_call.message.path == "queue":
            return self.resolve_future(
                credentials=api_call.credentials, uid=api_call.message.kwargs["uid"]
            )

        if api_call.message.path == "metadata":
            return self.metadata

        result = None
        is_blocking = api_call.message.blocking

        if is_blocking or self.is_subprocess:
            credentials: SyftVerifyKey = api_call.credentials
            api_call = api_call.message

            role = self.get_role_for_credentials(credentials=credentials)
            context = AuthedServiceContext(
                node=self, credentials=credentials, role=role, job_id=job_id
            )
            AuthNodeContextRegistry.set_node_context(self.id, context, credentials)

            user_config_registry = UserServiceConfigRegistry.from_role(role)

            if api_call.path not in user_config_registry:
                if ServiceConfigRegistry.path_exists(api_call.path):
                    return SyftError(
                        message=f"As a `{role}`,"
                        f"you have has no access to: {api_call.path}"
                    )  # type: ignore
                else:
                    return SyftError(
                        message=f"API call not in registered services: {api_call.path}"
                    )  # type: ignore

            _private_api_path = user_config_registry.private_path_for(api_call.path)
            method = self.get_service_method(_private_api_path)
            try:
                result = method(context, *api_call.args, **api_call.kwargs)
            except PySyftException as e:
                return e.handle()
            except Exception:
                result = SyftError(
                    message=f"Exception calling {api_call.path}. {traceback.format_exc()}"
                )
        else:
            return self.add_api_call_to_queue(api_call)
        return result

    def add_action_to_queue(
        self,
        action,
        credentials,
        parent_job_id=None,
        has_execute_permissions: bool = False,
        worker_pool_name: Optional[str] = None,
    ):
        job_id = UID()
        task_uid = UID()
        worker_settings = WorkerSettings.from_node(node=self)

        # Extract worker pool id from user code
        if action.user_code_id is not None:
            result = self.user_code_stash.get_by_uid(
                credentials=credentials, uid=action.user_code_id
            )

            # If result is Ok, then user code object exists
            if result.is_ok() and result.ok() is not None:
                user_code = result.ok()
                worker_pool_name = user_code.worker_pool_name

        # If worker pool id is not set, then use default worker pool
        # Else, get the worker pool for given uid
        if worker_pool_name is None:
            worker_pool = self.get_default_worker_pool()
        else:
            result = self.pool_stash.get_by_name(credentials, worker_pool_name)
            if result.is_err():
                return SyftError(message=f"{result.err()}")
            worker_pool = result.ok()

        # Create a Worker pool reference object
        worker_pool_ref = LinkedObject.from_obj(
            worker_pool,
            service_type=SyftWorkerPoolService,
            node_uid=self.id,
        )

        queue_item = ActionQueueItem(
            id=task_uid,
            node_uid=self.id,
            syft_client_verify_key=credentials,
            syft_node_location=self.id,
            job_id=job_id,
            worker_settings=worker_settings,
            args=[],
            kwargs={"action": action},
            has_execute_permissions=has_execute_permissions,
            worker_pool=worker_pool_ref,  # set worker pool reference as part of queue item
        )
        return self.add_queueitem_to_queue(
            queue_item, credentials, action, parent_job_id
        )

    def add_queueitem_to_queue(
        self, queue_item, credentials, action=None, parent_job_id=None
    ):
        log_id = UID()
        role = self.get_role_for_credentials(credentials=credentials)
        context = AuthedServiceContext(node=self, credentials=credentials, role=role)

        result_obj = ActionObject.empty()
        if action is not None:
            result_obj = ActionObject.obj_not_ready(id=action.result_id)
            result_obj.id = action.result_id
            result_obj.syft_resolved = False
            result_obj.syft_node_location = self.id
            result_obj.syft_client_verify_key = credentials

            action_service = self.get_service("actionservice")

            if not action_service.store.exists(uid=action.result_id):
                result = action_service.set_result_to_store(
                    result_action_object=result_obj,
                    context=context,
                )
                if result.is_err():
                    return result.err()

        job = Job(
            id=queue_item.job_id,
            result=result_obj,
            node_uid=self.id,
            syft_client_verify_key=credentials,
            syft_node_location=self.id,
            log_id=log_id,
            parent_job_id=parent_job_id,
            action=action,
        )

        # 🟡 TODO 36: Needs distributed lock
        self.queue_stash.set_placeholder(credentials, queue_item)
        self.job_stash.set(credentials, job)

        log_service = self.get_service("logservice")

        result = log_service.add(context, log_id)
        if isinstance(result, SyftError):
            return result
        return job

    def _get_existing_user_code_jobs(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> Union[List[Job], SyftError]:
        job_service = self.get_service("jobservice")
        return job_service.get_by_user_code_id(
            context=context, user_code_id=user_code_id
        )

    def _is_usercode_call_on_owned_kwargs(
        self, context: AuthedServiceContext, api_call: SyftAPICall
    ) -> bool:
        if api_call.path != "code.call":
            return False
        user_code_service = self.get_service("usercodeservice")
        return user_code_service.is_execution_on_owned_args(api_call.kwargs, context)

    def add_api_call_to_queue(self, api_call, parent_job_id=None):
        unsigned_call = api_call
        if isinstance(api_call, SignedSyftAPICall):
            unsigned_call = api_call.message

        credentials = api_call.credentials
        context = AuthedServiceContext(
            node=self,
            credentials=credentials,
            role=self.get_role_for_credentials(credentials=credentials),
        )

        is_user_code = unsigned_call.path == "code.call"

        service_str, method_str = unsigned_call.path.split(".")

        action = None
        if is_user_code:
            action = Action.from_api_call(unsigned_call)

            is_usercode_call_on_owned_kwargs = self._is_usercode_call_on_owned_kwargs(
                context, unsigned_call
            )
            # Low side does not execute jobs, unless this is a mock execution
            if (
                not is_usercode_call_on_owned_kwargs
                and self.node_side_type == NodeSideType.LOW_SIDE
            ):
                existing_jobs = self._get_existing_user_code_jobs(
                    context, action.user_code_id
                )
                if isinstance(existing_jobs, SyftError):
                    return existing_jobs
                elif len(existing_jobs) > 0:
                    # Print warning if there are existing jobs for this user code
                    # relative
                    from ..util.util import prompt_warning_message

                    prompt_warning_message(
                        "There are existing jobs for this user code, returning the latest one"
                    )
                    return existing_jobs[-1]
                else:
                    return SyftError(
                        message="Please wait for the admin to allow the execution of this code"
                    )

            return self.add_action_to_queue(
                action, api_call.credentials, parent_job_id=parent_job_id
            )

        else:
            worker_settings = WorkerSettings.from_node(node=self)
            default_worker_pool = self.get_default_worker_pool()
            worker_pool = LinkedObject.from_obj(
                default_worker_pool,
                service_type=SyftWorkerPoolService,
                node_uid=self.id,
            )
            queue_item = QueueItem(
                id=UID(),
                node_uid=self.id,
                syft_client_verify_key=api_call.credentials,
                syft_node_location=self.id,
                job_id=UID(),
                worker_settings=worker_settings,
                service=service_str,
                method=method_str,
                args=unsigned_call.args,
                kwargs=unsigned_call.kwargs,
                worker_pool=worker_pool,
            )
            return self.add_queueitem_to_queue(
                queue_item,
                api_call.credentials,
                action=None,
                parent_job_id=parent_job_id,
            )

    @property
    def pool_stash(self) -> SyftWorkerPoolStash:
        return self.get_service(SyftWorkerPoolService).stash

    @property
    def user_code_stash(self) -> UserCodeStash:
        return self.get_service(UserCodeService).stash

    def get_default_worker_pool(self):
        result = self.pool_stash.get_by_name(
            credentials=self.verify_key,
            pool_name=get_default_worker_pool_name(),
        )
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        worker_pool = result.ok()
        return worker_pool

    def get_api(
        self,
        for_user: Optional[SyftVerifyKey] = None,
        communication_protocol: Optional[PROTOCOL_TYPE] = None,
    ) -> SyftAPI:
        return SyftAPI.for_user(
            node=self,
            user_verify_key=for_user,
            communication_protocol=communication_protocol,
        )

    def get_method_with_context(
        self, function: Callable, context: NodeServiceContext
    ) -> Callable:
        method = self.get_service_method(function)
        return partial(method, context=context)

    def get_unauthed_context(
        self, login_credentials: UserLoginCredentials
    ) -> NodeServiceContext:
        return UnauthedServiceContext(node=self, login_credentials=login_credentials)

    def create_initial_settings(self, admin_email: str) -> Optional[NodeSettingsV2]:
        if self.name is None:
            self.name = random_name()
        try:
            settings_stash = SettingsStash(store=self.document_store)
            settings_exists = settings_stash.get_all(self.signing_key.verify_key).ok()
            if settings_exists:
                self.name = settings_exists[0].name
                return None
            else:
                # Currently we allow automatic user registration on enclaves,
                # as enclaves do not have superusers
                if self.node_type == NodeType.ENCLAVE:
                    flags.CAN_REGISTER = True
                new_settings = NodeSettingsV2(
                    id=self.id,
                    name=self.name,
                    verify_key=self.verify_key,
                    node_type=self.node_type,
                    deployed_on=datetime.now().date().strftime("%m/%d/%Y"),
                    signup_enabled=flags.CAN_REGISTER,
                    admin_email=admin_email,
                    node_side_type=self.node_side_type.value,
                    show_warnings=self.enable_warnings,
                )
                result = settings_stash.set(
                    credentials=self.signing_key.verify_key, settings=new_settings
                )
                if result.is_ok():
                    return result.ok()
                return None
        except Exception as e:
            print("create_worker_metadata failed", e)


def create_admin_new(
    name: str,
    email: str,
    password: str,
    node: AbstractNode,
) -> Optional[User]:
    try:
        user_stash = UserStash(store=node.document_store)
        row_exists = user_stash.get_by_email(
            credentials=node.signing_key.verify_key, email=email
        ).ok()
        if row_exists:
            return None
        else:
            create_user = UserCreate(
                name=name,
                email=email,
                password=password,
                password_verify=password,
                role=ServiceRole.ADMIN,
            )
            # New User Initialization
            # 🟡 TODO: change later but for now this gives the main user super user automatically
            user = create_user.to(User)
            user.signing_key = node.signing_key
            user.verify_key = user.signing_key.verify_key
            result = user_stash.set(
                credentials=node.signing_key.verify_key,
                user=user,
                ignore_duplicates=True,
            )
            if result.is_ok():
                return result.ok()
            else:
                raise Exception(f"Could not create user: {result}")
    except Exception as e:
        print("Unable to create new admin", e)


def create_oblv_key_pair(
    worker: Node,
) -> Optional[str]:
    try:
        # relative
        from ..external.oblv.oblv_keys_stash import OblvKeys
        from ..external.oblv.oblv_keys_stash import OblvKeysStash
        from ..external.oblv.oblv_service import generate_oblv_key

        oblv_keys_stash = OblvKeysStash(store=worker.document_store)

        if not len(oblv_keys_stash):
            public_key, private_key = generate_oblv_key(oblv_key_name=worker.name)
            oblv_keys = OblvKeys(public_key=public_key, private_key=private_key)
            res = oblv_keys_stash.set(worker.signing_key.verify_key, oblv_keys)
            if res.is_ok():
                print("Successfully generated Oblv Key pair at startup")
            return res.err()
        else:
            print(f"Using Existing Public/Private Key pair: {len(oblv_keys_stash)}")
    except Exception as e:
        print("Unable to create Oblv Keys.", e)


class NodeRegistry:
    __node_registry__: Dict[UID, Node] = {}

    @classmethod
    def set_node_for(
        cls,
        node_uid: Union[UID, str],
        node: Node,
    ) -> None:
        if isinstance(node_uid, str):
            node_uid = UID.from_string(node_uid)

        cls.__node_registry__[node_uid] = node

    @classmethod
    def node_for(cls, node_uid: UID) -> Node:
        return cls.__node_registry__.get(node_uid, None)

    @classmethod
    def get_all_nodes(cls) -> List[Node]:
        return list(cls.__node_registry__.values())


def get_default_worker_tag_by_env(dev_mode=False):
    if in_kubernetes():
        return get_default_worker_image()
    elif dev_mode:
        return "local-dev"
    else:
        return __version__


def create_default_worker_pool(node: Node) -> Optional[SyftError]:
    credentials = node.verify_key
    pull_image = not node.dev_mode
    image_stash = node.get_service(SyftWorkerImageService).stash
    default_pool_name = get_default_worker_pool_name()
    default_worker_pool = node.get_default_worker_pool()
    default_worker_tag = get_default_worker_tag_by_env(node.dev_mode)
    worker_count = get_default_worker_pool_count()
    context = AuthedServiceContext(
        node=node,
        credentials=credentials,
        role=ServiceRole.ADMIN,
    )

    print(f"Creating default worker image with tag='{default_worker_tag}'")
    # Get/Create a default worker SyftWorkerImage
    default_image = create_default_image(
        credentials=credentials,
        image_stash=image_stash,
        tag=default_worker_tag,
        in_kubernetes=in_kubernetes(),
    )
    if isinstance(default_image, SyftError):
        print("Failed to create default worker image: ", default_image.message)
        return default_image

    if not default_image.is_built:
        print(f"Building default worker image with tag={default_worker_tag}")
        image_build_method = node.get_service_method(SyftWorkerImageService.build)
        # Build the Image for given tag
        result = image_build_method(
            context,
            image_uid=default_image.id,
            tag=DEFAULT_WORKER_IMAGE_TAG,
            pull=pull_image,
        )

        if isinstance(result, SyftError):
            print("Failed to build default worker image: ", result.message)
            return

    # Create worker pool if it doesn't exists
    print(
        "Setting up worker pool"
        f"name={default_pool_name} "
        f"workers={worker_count} "
        f"image_uid={default_image.id} "
        f"in_memory={node.in_memory_workers}"
    )
    if default_worker_pool is None:
        worker_to_add_ = worker_count
        create_pool_method = node.get_service_method(SyftWorkerPoolService.launch)
        result = create_pool_method(
            context,
            name=default_pool_name,
            image_uid=default_image.id,
            num_workers=worker_count,
        )
    else:
        # Else add a worker to existing worker pool
        worker_to_add_ = max(default_worker_pool.max_count, worker_count) - len(
            default_worker_pool.worker_list
        )
        add_worker_method = node.get_service_method(SyftWorkerPoolService.add_workers)
        result = add_worker_method(
            context=context,
            number=worker_to_add_,
            pool_name=default_pool_name,
        )

    if isinstance(result, SyftError):
        print(f"Default worker pool error. {result.message}")
        return

    for n in range(worker_to_add_):
        container_status = result[n]
        if container_status.error:
            print(
                f"Failed to create container: Worker: {container_status.worker},"
                f"Error: {container_status.error}"
            )
            return

    print("Created default worker pool.")
