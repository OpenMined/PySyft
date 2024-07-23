# future
from __future__ import annotations

# stdlib
from collections import OrderedDict
from collections.abc import Callable
from datetime import MINYEAR
from datetime import datetime
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess  # nosec
import sys
from time import sleep
import traceback
from typing import Any
from typing import cast

# third party
from nacl.signing import SigningKey
from result import Err
from result import Result

# relative
from .. import __version__
from ..abstract_server import AbstractServer
from ..abstract_server import ServerSideType
from ..abstract_server import ServerType
from ..client.api import SignedSyftAPICall
from ..client.api import SyftAPI
from ..client.api import SyftAPICall
from ..client.api import SyftAPIData
from ..client.api import debox_signed_syftapicall_response
from ..client.client import SyftClient
from ..exceptions.exception import PySyftException
from ..protocol.data_protocol import PROTOCOL_TYPE
from ..protocol.data_protocol import get_data_protocol
from ..service.action.action_object import Action
from ..service.action.action_object import ActionObject
from ..service.action.action_store import ActionStore
from ..service.action.action_store import DictActionStore
from ..service.action.action_store import MongoActionStore
from ..service.action.action_store import SQLiteActionStore
from ..service.blob_storage.service import BlobStorageService
from ..service.code.user_code_service import UserCodeService
from ..service.code.user_code_stash import UserCodeStash
from ..service.context import AuthedServiceContext
from ..service.context import ServerServiceContext
from ..service.context import UnauthedServiceContext
from ..service.context import UserLoginCredentials
from ..service.job.job_stash import Job
from ..service.job.job_stash import JobStash
from ..service.job.job_stash import JobStatus
from ..service.job.job_stash import JobType
from ..service.metadata.server_metadata import ServerMetadata
from ..service.network.network_service import NetworkService
from ..service.network.utils import PeerHealthCheckTask
from ..service.notifier.notifier_service import NotifierService
from ..service.queue.base_queue import AbstractMessageHandler
from ..service.queue.base_queue import QueueConsumer
from ..service.queue.base_queue import QueueProducer
from ..service.queue.queue import APICallMessageHandler
from ..service.queue.queue import QueueManager
from ..service.queue.queue_stash import APIEndpointQueueItem
from ..service.queue.queue_stash import ActionQueueItem
from ..service.queue.queue_stash import QueueItem
from ..service.queue.queue_stash import QueueStash
from ..service.queue.zmq_queue import QueueConfig
from ..service.queue.zmq_queue import ZMQClientConfig
from ..service.queue.zmq_queue import ZMQQueueConfig
from ..service.response import SyftError
from ..service.service import AbstractService
from ..service.service import ServiceConfigRegistry
from ..service.service import UserServiceConfigRegistry
from ..service.settings.settings import ServerSettings
from ..service.settings.settings import ServerSettingsUpdate
from ..service.settings.settings_stash import SettingsStash
from ..service.user.user import User
from ..service.user.user import UserCreate
from ..service.user.user import UserView
from ..service.user.user_roles import ServiceRole
from ..service.user.user_service import UserService
from ..service.user.user_stash import UserStash
from ..service.worker.utils import DEFAULT_WORKER_IMAGE_TAG
from ..service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ..service.worker.utils import create_default_image
from ..service.worker.worker_image_service import SyftWorkerImageService
from ..service.worker.worker_pool import WorkerPool
from ..service.worker.worker_pool_service import SyftWorkerPoolService
from ..service.worker.worker_pool_stash import SyftWorkerPoolStash
from ..service.worker.worker_stash import WorkerStash
from ..store.blob_storage import BlobStorageConfig
from ..store.blob_storage.on_disk import OnDiskBlobStorageClientConfig
from ..store.blob_storage.on_disk import OnDiskBlobStorageConfig
from ..store.blob_storage.seaweedfs import SeaweedFSBlobDeposit
from ..store.dict_document_store import DictStoreConfig
from ..store.document_store import StoreConfig
from ..store.linked_obj import LinkedObject
from ..store.mongo_document_store import MongoStoreConfig
from ..store.sqlite_document_store import SQLiteStoreClientConfig
from ..store.sqlite_document_store import SQLiteStoreConfig
from ..types.datetime import DATETIME_FORMAT
from ..types.syft_metaclass import Empty
from ..types.syft_object import Context
from ..types.syft_object import PartialSyftObject
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID
from ..util.experimental_flags import flags
from ..util.telemetry import instrument
from ..util.util import get_dev_mode
from ..util.util import get_env
from ..util.util import get_queue_address
from ..util.util import random_name
from ..util.util import str_to_bool
from ..util.util import thread_ident
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .service_registry import ServiceRegistry
from .utils import get_named_server_uid
from .utils import get_temp_dir_for_server
from .utils import remove_temp_dir_for_server
from .worker_settings import WorkerSettings

logger = logging.getLogger(__name__)

# if user code needs to be serded and its not available we can call this to refresh
# the code for a specific server UID and thread
CODE_RELOADER: dict[int, Callable] = {}


SERVER_PRIVATE_KEY = "SERVER_PRIVATE_KEY"
SERVER_UID = "SERVER_UID"
SERVER_TYPE = "SERVER_TYPE"
SERVER_NAME = "SERVER_NAME"
SERVER_SIDE_TYPE = "SERVER_SIDE_TYPE"

DEFAULT_ROOT_EMAIL = "DEFAULT_ROOT_EMAIL"
DEFAULT_ROOT_USERNAME = "DEFAULT_ROOT_USERNAME"
DEFAULT_ROOT_PASSWORD = "DEFAULT_ROOT_PASSWORD"  # nosec


def get_private_key_env() -> str | None:
    return get_env(SERVER_PRIVATE_KEY)


def get_server_type() -> str | None:
    return get_env(SERVER_TYPE, "datasite")


def get_server_name() -> str | None:
    return get_env(SERVER_NAME, None)


def get_server_side_type() -> str | None:
    return get_env(SERVER_SIDE_TYPE, "high")


def get_server_uid_env() -> str | None:
    return get_env(SERVER_UID)


def get_default_root_email() -> str | None:
    return get_env(DEFAULT_ROOT_EMAIL, "info@openmined.org")


def get_default_root_username() -> str | None:
    return get_env(DEFAULT_ROOT_USERNAME, "Jane Doe")


def get_default_root_password() -> str | None:
    return get_env(DEFAULT_ROOT_PASSWORD, "changethis")  # nosec


def get_enable_warnings() -> bool:
    return str_to_bool(get_env("ENABLE_WARNINGS", "False"))


def get_container_host() -> str | None:
    return get_env("CONTAINER_HOST")


def get_default_worker_image() -> str | None:
    return get_env("DEFAULT_WORKER_POOL_IMAGE")


def get_default_worker_pool_name() -> str | None:
    return get_env("DEFAULT_WORKER_POOL_NAME", DEFAULT_WORKER_POOL_NAME)


def get_default_bucket_name() -> str:
    env = get_env("DEFAULT_BUCKET_NAME")
    server_id = get_server_uid_env() or "syft-bucket"
    return env or server_id or "syft-bucket"


def get_default_worker_pool_count(server: Server) -> int:
    return int(
        get_env(
            "DEFAULT_WORKER_POOL_COUNT", server.queue_config.client_config.n_consumers
        )
    )


def get_default_worker_pool_pod_annotations() -> dict[str, str] | None:
    annotations = get_env("DEFAULT_WORKER_POOL_POD_ANNOTATIONS", "null")
    return json.loads(annotations)


def get_default_worker_pool_pod_labels() -> dict[str, str] | None:
    labels = get_env("DEFAULT_WORKER_POOL_POD_LABELS", "null")
    return json.loads(labels)


def in_kubernetes() -> bool:
    return get_container_host() == "k8s"


def get_venv_packages() -> str:
    try:
        # subprocess call is safe because it uses a fully qualified path and fixed arguments
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],  # nosec
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.stderr}"


def get_syft_worker() -> bool:
    return str_to_bool(get_env("SYFT_WORKER", "false"))


def get_k8s_pod_name() -> str | None:
    return get_env("K8S_POD_NAME")


def get_syft_worker_uid() -> str | None:
    is_worker = get_syft_worker()
    pod_name = get_k8s_pod_name()
    uid = get_env("SYFT_WORKER_UID")
    # if uid is empty is a K8S worker, generate a uid from the pod name
    if (not uid) and is_worker and pod_name:
        uid = str(UID.with_seed(pod_name))
    return uid


signing_key_env = get_private_key_env()
server_uid_env = get_server_uid_env()

default_root_email = get_default_root_email()
default_root_username = get_default_root_username()
default_root_password = get_default_root_password()


class AuthServerContextRegistry:
    __server_context_registry__: dict[str, ServerServiceContext] = OrderedDict()

    @classmethod
    def set_server_context(
        cls,
        server_uid: UID | str,
        context: ServerServiceContext,
        user_verify_key: SyftVerifyKey | str,
    ) -> None:
        if isinstance(server_uid, str):
            server_uid = UID.from_string(server_uid)

        if isinstance(user_verify_key, str):
            user_verify_key = SyftVerifyKey.from_string(user_verify_key)

        key = cls._get_key(server_uid=server_uid, user_verify_key=user_verify_key)

        cls.__server_context_registry__[key] = context

    @staticmethod
    def _get_key(server_uid: UID, user_verify_key: SyftVerifyKey) -> str:
        return "-".join(str(x) for x in (server_uid, user_verify_key))

    @classmethod
    def auth_context_for_user(
        cls,
        server_uid: UID,
        user_verify_key: SyftVerifyKey,
    ) -> AuthedServiceContext | None:
        key = cls._get_key(server_uid=server_uid, user_verify_key=user_verify_key)
        return cls.__server_context_registry__.get(key)


@instrument
class Server(AbstractServer):
    signing_key: SyftSigningKey | None
    required_signed_calls: bool = True
    packages: str

    def __init__(
        self,
        *,  # Trasterisk
        name: str | None = None,
        id: UID | None = None,
        signing_key: SyftSigningKey | SigningKey | None = None,
        action_store_config: StoreConfig | None = None,
        document_store_config: StoreConfig | None = None,
        root_email: str | None = default_root_email,
        root_username: str | None = default_root_username,
        root_password: str | None = default_root_password,
        processes: int = 0,
        is_subprocess: bool = False,
        server_type: str | ServerType = ServerType.DATASITE,
        local_db: bool = False,
        reset: bool = False,
        blob_storage_config: BlobStorageConfig | None = None,
        queue_config: QueueConfig | None = None,
        queue_port: int | None = None,
        n_consumers: int = 0,
        create_producer: bool = False,
        thread_workers: bool = False,
        server_side_type: str | ServerSideType = ServerSideType.HIGH_SIDE,
        enable_warnings: bool = False,
        dev_mode: bool = False,
        migrate: bool = False,
        in_memory_workers: bool = True,
        smtp_username: str | None = None,
        smtp_password: str | None = None,
        email_sender: str | None = None,
        smtp_port: int | None = None,
        smtp_host: str | None = None,
        association_request_auto_approval: bool = False,
        background_tasks: bool = False,
    ):
        # 🟡 TODO 22: change our ENV variable format and default init args to make this
        # less horrible or add some convenience functions
        self.dev_mode = dev_mode or get_dev_mode()
        self.id = UID.from_string(server_uid_env) if server_uid_env else (id or UID())
        self.packages = ""
        self.processes = processes
        self.is_subprocess = is_subprocess
        self.name = name or random_name()
        self.enable_warnings = enable_warnings
        self.in_memory_workers = in_memory_workers
        self.server_type = ServerType(server_type)
        self.server_side_type = ServerSideType(server_side_type)
        self.client_cache: dict = {}
        self.peer_client_cache: dict = {}

        if isinstance(server_type, str):
            server_type = ServerType(server_type)
        self.server_type = server_type

        if isinstance(server_side_type, str):
            server_side_type = ServerSideType(server_side_type)
        self.server_side_type = server_side_type

        skey = None
        if signing_key_env:
            skey = SyftSigningKey.from_string(signing_key_env)
        elif isinstance(signing_key, SigningKey):
            skey = SyftSigningKey(signing_key=signing_key)
        else:
            skey = signing_key
        self.signing_key = skey or SyftSigningKey.generate()

        self.association_request_auto_approval = association_request_auto_approval

        self.queue_config = self.create_queue_config(
            n_consumers=n_consumers,
            create_producer=create_producer,
            thread_workers=thread_workers,
            queue_port=queue_port,
            queue_config=queue_config,
        )

        # must call before initializing stores
        if reset:
            self.remove_temp_dir()

        use_sqlite = local_db or (processes > 0 and not is_subprocess)
        document_store_config = document_store_config or self.get_default_store(
            use_sqlite=use_sqlite,
            store_type="Document Store",
        )
        action_store_config = action_store_config or self.get_default_store(
            use_sqlite=use_sqlite,
            store_type="Action Store",
        )
        self.init_stores(
            action_store_config=action_store_config,
            document_store_config=document_store_config,
        )

        # construct services only after init stores
        self.services: ServiceRegistry = ServiceRegistry.for_server(self)

        create_admin_new(  # nosec B106
            name=root_username,
            email=root_email,
            password=root_password,
            server=self,
        )

        NotifierService.init_notifier(
            server=self,
            email_password=smtp_password,
            email_username=smtp_username,
            email_sender=email_sender,
            smtp_port=smtp_port,
            smtp_host=smtp_host,
        )

        self.post_init()

        if migrate:
            self.find_and_migrate_data()
        else:
            self.find_and_migrate_data([ServerSettings])

        self.create_initial_settings(admin_email=root_email)

        self.init_blob_storage(config=blob_storage_config)

        # Migrate data before any operation on db

        # first migrate, for backwards compatibility
        self.init_queue_manager(queue_config=self.queue_config)

        context = AuthedServiceContext(
            server=self,
            credentials=self.verify_key,
            role=ServiceRole.ADMIN,
        )

        self.peer_health_manager: PeerHealthCheckTask | None = None
        if background_tasks:
            self.run_peer_health_checks(context=context)

        ServerRegistry.set_server_for(self.id, self)

    @property
    def runs_in_docker(self) -> bool:
        path = "/proc/self/cgroup"
        return (
            os.path.exists("/.dockerenv")
            or os.path.isfile(path)
            and any("docker" in line for line in open(path))
        )

    def get_default_store(self, use_sqlite: bool, store_type: str) -> StoreConfig:
        if use_sqlite:
            path = self.get_temp_dir("db")
            file_name: str = f"{self.id}.sqlite"
            if self.dev_mode:
                logger.debug(f"{store_type}'s SQLite DB path: {path/file_name}")
            return SQLiteStoreConfig(
                client_config=SQLiteStoreClientConfig(
                    filename=file_name,
                    path=path,
                )
            )
        return DictStoreConfig()

    def init_blob_storage(self, config: BlobStorageConfig | None = None) -> None:
        if config is None:
            client_config = OnDiskBlobStorageClientConfig(
                base_directory=self.get_temp_dir("blob")
            )
            config_ = OnDiskBlobStorageConfig(
                client_config=client_config,
                min_blob_size=os.getenv("MIN_SIZE_BLOB_STORAGE_MB", 16),
            )
        else:
            config_ = config
        self.blob_store_config = config_
        self.blob_storage_client = config_.client_type(config=config_.client_config)

        # relative
        from ..store.blob_storage.seaweedfs import SeaweedFSConfig

        if isinstance(config, SeaweedFSConfig) and self.signing_key:
            blob_storage_service = self.get_service(BlobStorageService)
            remote_profiles = blob_storage_service.remote_profile_stash.get_all(
                credentials=self.signing_key.verify_key, has_permission=True
            ).ok()
            for remote_profile in remote_profiles:
                self.blob_store_config.client_config.remote_profiles[
                    remote_profile.profile_name
                ] = remote_profile

        if self.dev_mode:
            if isinstance(self.blob_store_config, OnDiskBlobStorageConfig):
                logger.debug(
                    f"Using on-disk blob storage with path: "
                    f"{self.blob_store_config.client_config.base_directory}",
                )
            logger.debug(
                f"Minimum object size to be saved to the blob storage: "
                f"{self.blob_store_config.min_blob_size} (MB)."
            )

    def run_peer_health_checks(self, context: AuthedServiceContext) -> None:
        self.peer_health_manager = PeerHealthCheckTask()
        self.peer_health_manager.run(context=context)

    def stop(self) -> None:
        if self.peer_health_manager is not None:
            self.peer_health_manager.stop()

        for consumer_list in self.queue_manager.consumers.values():
            for c in consumer_list:
                c.close()
        for p in self.queue_manager.producers.values():
            p.close()

        self.queue_manager.producers.clear()
        self.queue_manager.consumers.clear()

        ServerRegistry.remove_server(self.id)

    def close(self) -> None:
        self.stop()

    def cleanup(self) -> None:
        self.stop()
        self.remove_temp_dir()

    def create_queue_config(
        self,
        n_consumers: int,
        create_producer: bool,
        thread_workers: bool,
        queue_port: int | None,
        queue_config: QueueConfig | None,
    ) -> QueueConfig:
        if queue_config:
            queue_config_ = queue_config
        elif queue_port is not None or n_consumers > 0 or create_producer:
            if not create_producer and queue_port is None:
                logger.warn("No queue port defined to bind consumers.")
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

    def init_queue_manager(self, queue_config: QueueConfig) -> None:
        MessageHandlers = [APICallMessageHandler]
        if self.is_subprocess:
            return None

        self.queue_manager = QueueManager(config=queue_config)
        for message_handler in MessageHandlers:
            queue_name = message_handler.queue_name
            # client config
            if getattr(queue_config.client_config, "create_producer", True):
                context = AuthedServiceContext(
                    server=self,
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
                logger.info(
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
        message_handler: type[AbstractMessageHandler] = APICallMessageHandler,
    ) -> None:
        consumer: QueueConsumer = self.queue_manager.create_consumer(
            message_handler,
            address=address,
            service_name=service_name,
            worker_stash=self.worker_stash,
            syft_worker_id=syft_worker_id,
        )
        consumer.run()

    def remove_consumer_with_id(self, syft_worker_id: UID) -> None:
        for consumers in self.queue_manager.consumers.values():
            # Grab the list of consumers for the given queue
            consumer_to_pop = None
            for consumer_idx, consumer in enumerate(consumers):
                if consumer.syft_worker_id == syft_worker_id:
                    consumer.close()
                    consumer_to_pop = consumer_idx
                    break
            if consumer_to_pop is not None:
                consumers.pop(consumer_to_pop)

    @classmethod
    def named(
        cls: type[Server],
        *,  # Trasterisk
        name: str,
        processes: int = 0,
        reset: bool = False,
        local_db: bool = False,
        server_type: str | ServerType = ServerType.DATASITE,
        server_side_type: str | ServerSideType = ServerSideType.HIGH_SIDE,
        enable_warnings: bool = False,
        n_consumers: int = 0,
        thread_workers: bool = False,
        create_producer: bool = False,
        queue_port: int | None = None,
        dev_mode: bool = False,
        migrate: bool = False,
        in_memory_workers: bool = True,
        association_request_auto_approval: bool = False,
        background_tasks: bool = False,
    ) -> Server:
        uid = get_named_server_uid(name)
        name_hash = hashlib.sha256(name.encode("utf8")).digest()
        key = SyftSigningKey(signing_key=SigningKey(name_hash))
        blob_storage_config = None

        server_type = ServerType(server_type)
        server_side_type = ServerSideType(server_side_type)

        return cls(
            name=name,
            id=uid,
            signing_key=key,
            processes=processes,
            local_db=local_db,
            server_type=server_type,
            server_side_type=server_side_type,
            enable_warnings=enable_warnings,
            blob_storage_config=blob_storage_config,
            queue_port=queue_port,
            n_consumers=n_consumers,
            thread_workers=thread_workers,
            create_producer=create_producer,
            dev_mode=dev_mode,
            migrate=migrate,
            in_memory_workers=in_memory_workers,
            reset=reset,
            association_request_auto_approval=association_request_auto_approval,
            background_tasks=background_tasks,
        )

    def is_root(self, credentials: SyftVerifyKey) -> bool:
        return credentials == self.verify_key

    @property
    def root_client(self) -> SyftClient:
        # relative
        from ..client.client import PythonConnection

        connection = PythonConnection(server=self)
        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type
        root_client = client_type(connection=connection, credentials=self.signing_key)
        if root_client.api.refresh_api_callback is not None:
            root_client.api.refresh_api_callback()
        return root_client

    def _find_klasses_pending_for_migration(
        self, object_types: list[SyftObject]
    ) -> list[SyftObject]:
        context = AuthedServiceContext(
            server=self,
            credentials=self.verify_key,
            role=ServiceRole.ADMIN,
        )
        migration_state_service = self.services.migration

        klasses_to_be_migrated = []

        for object_type in object_types:
            canonical_name = object_type.__canonical_name__
            object_version = object_type.__version__

            migration_state = migration_state_service.get_state(context, canonical_name)
            if isinstance(migration_state, SyftError):
                raise Exception(
                    f"Failed to get migration state for {canonical_name}. Error: {migration_state}"
                )
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

    def find_and_migrate_data(
        self, document_store_object_types: list[type[SyftObject]] | None = None
    ) -> None:
        context = AuthedServiceContext(
            server=self,
            credentials=self.verify_key,
            role=ServiceRole.ADMIN,
        )
        migration_service = self.get_service("migrationservice")
        return migration_service.migrate_data(context, document_store_object_types)

    @property
    def guest_client(self) -> SyftClient:
        return self.get_guest_client()

    @property
    def current_protocol(self) -> str | int:
        data_protocol = get_data_protocol()
        return data_protocol.latest_version

    def get_guest_client(self, verbose: bool = True) -> SyftClient:
        # relative
        from ..client.client import PythonConnection

        connection = PythonConnection(server=self)
        if verbose and self.server_side_type:
            message: str = (
                f"Logged into <{self.name}: {self.server_side_type.value.capitalize()} "
            )
            if self.server_type:
                message += f"side {self.server_type.value.capitalize()} > as GUEST"
            logger.debug(message)

        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type

        guest_client = client_type(
            connection=connection, credentials=SyftSigningKey.generate()
        )
        if guest_client.api.refresh_api_callback is not None:
            guest_client.api.refresh_api_callback()
        return guest_client

    def __repr__(self) -> str:
        service_string = ""
        if not self.is_subprocess:
            services = [service.__name__ for service in self.services]
            service_string = ", ".join(sorted(services))
            service_string = f"\n\nServices:\n{service_string}"
        return f"{type(self).__name__}: {self.name} - {self.id} - {self.server_type}{service_string}"

    def post_init(self) -> None:
        context = AuthedServiceContext(
            server=self, credentials=self.verify_key, role=ServiceRole.ADMIN
        )
        AuthServerContextRegistry.set_server_context(
            server_uid=self.id, user_verify_key=self.verify_key, context=context
        )

        if "usercodeservice" in self.service_path_map:
            user_code_service = self.get_service(UserCodeService)
            user_code_service.load_user_code(context=context)

        def reload_user_code() -> None:
            user_code_service.load_user_code(context=context)

        ti = thread_ident()
        if ti is not None:
            CODE_RELOADER[ti] = reload_user_code

    def init_stores(
        self,
        document_store_config: StoreConfig,
        action_store_config: StoreConfig,
    ) -> None:
        # We add the python id of the current server in order
        # to create one connection per Server object in MongoClientCache
        # so that we avoid closing the connection from a
        # different thread through the garbage collection
        if isinstance(document_store_config, MongoStoreConfig):
            document_store_config.client_config.server_obj_python_id = id(self)

        self.document_store_config = document_store_config
        self.document_store = document_store_config.store_type(
            server_uid=self.id,
            root_verify_key=self.verify_key,
            store_config=document_store_config,
        )

        if isinstance(action_store_config, SQLiteStoreConfig):
            self.action_store: ActionStore = SQLiteActionStore(
                server_uid=self.id,
                store_config=action_store_config,
                root_verify_key=self.verify_key,
                document_store=self.document_store,
            )
        elif isinstance(action_store_config, MongoStoreConfig):
            # We add the python id of the current server in order
            # to create one connection per Server object in MongoClientCache
            # so that we avoid closing the connection from a
            # different thread through the garbage collection
            action_store_config.client_config.server_obj_python_id = id(self)

            self.action_store = MongoActionStore(
                server_uid=self.id,
                root_verify_key=self.verify_key,
                store_config=action_store_config,
                document_store=self.document_store,
            )
        else:
            self.action_store = DictActionStore(
                server_uid=self.id,
                root_verify_key=self.verify_key,
                document_store=self.document_store,
            )

        self.action_store_config = action_store_config
        self.queue_stash = QueueStash(store=self.document_store)

    @property
    def job_stash(self) -> JobStash:
        return self.get_service("jobservice").stash

    @property
    def worker_stash(self) -> WorkerStash:
        return self.get_service("workerservice").stash

    @property
    def service_path_map(self) -> dict[str, AbstractService]:
        return self.services.service_path_map

    @property
    def initialized_services(self) -> list[AbstractService]:
        return self.services.services

    def get_service_method(self, path_or_func: str | Callable) -> Callable:
        if callable(path_or_func):
            path_or_func = path_or_func.__qualname__
        return self._get_service_method_from_path(path_or_func)

    def get_service(self, path_or_func: str | Callable) -> AbstractService:
        return self.services.get_service(path_or_func)

    def _get_service_method_from_path(self, path: str) -> Callable:
        path_list = path.split(".")
        method_name = path_list.pop()
        service_obj = self.services._get_service_from_path(path=path)

        return getattr(service_obj, method_name)

    def get_temp_dir(self, dir_name: str = "") -> Path:
        """
        Get a temporary directory unique to the server.
        Provide all dbs, blob dirs, and locks using this directory.
        """
        return get_temp_dir_for_server(self.id, dir_name)

    def remove_temp_dir(self) -> None:
        """
        Remove the temporary directory for this server.
        """
        remove_temp_dir_for_server(self.id)

    def update_self(self, settings: ServerSettings) -> None:
        updateable_attrs = (
            ServerSettingsUpdate.model_fields.keys()
            - PartialSyftObject.model_fields.keys()
        )
        for attr_name in updateable_attrs:
            attr = getattr(settings, attr_name)
            if attr is not Empty:
                setattr(self, attr_name, attr)

    @property
    def settings(self) -> ServerSettings:
        settings_stash = SettingsStash(store=self.document_store)
        if self.signing_key is None:
            raise ValueError(f"{self} has no signing key")
        settings = settings_stash.get_all(self.signing_key.verify_key)
        if settings.is_err():
            raise ValueError(
                f"Cannot get server settings for '{self.name}'. Error: {settings.err()}"
            )
        if settings.is_ok() and len(settings.ok()) > 0:
            settings = settings.ok()[0]
        self.update_self(settings)
        return settings

    @property
    def metadata(self) -> ServerMetadata:
        show_warnings = self.enable_warnings
        settings_data = self.settings
        name = settings_data.name
        organization = settings_data.organization
        description = settings_data.description
        show_warnings = settings_data.show_warnings
        server_type = (
            settings_data.server_type.value if settings_data.server_type else ""
        )
        server_side_type = (
            settings_data.server_side_type.value
            if settings_data.server_side_type
            else ""
        )
        eager_execution_enabled = settings_data.eager_execution_enabled

        return ServerMetadata(
            name=name,
            id=self.id,
            verify_key=self.verify_key,
            highest_version=SYFT_OBJECT_VERSION_1,
            lowest_version=SYFT_OBJECT_VERSION_1,
            syft_version=__version__,
            description=description,
            organization=organization,
            server_type=server_type,
            server_side_type=server_side_type,
            show_warnings=show_warnings,
            eager_execution_enabled=eager_execution_enabled,
            min_size_blob_storage_mb=self.blob_store_config.min_blob_size,
        )

    @property
    def icon(self) -> str:
        return "🦾"

    @property
    def verify_key(self) -> SyftVerifyKey:
        if self.signing_key is None:
            raise ValueError(f"{self} has no signing key")
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
    ) -> QueueItem | None | SyftError:
        # stdlib

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
    ) -> QueueItem | None | SyftError:
        result = self.queue_stash.pop_on_complete(credentials, uid)

        if result.is_ok():
            queue_obj = result.ok()
            queue_obj._set_obj_location_(
                server_uid=self.id,
                credentials=credentials,
            )
            return queue_obj
        return result.err()

    def forward_message(
        self, api_call: SyftAPICall | SignedSyftAPICall
    ) -> Result | QueueItem | SyftObject | SyftError | Any:
        server_uid = api_call.message.server_uid
        if "networkservice" not in self.service_path_map:
            return SyftError(
                message=(
                    "Server has no network service so we can't "
                    f"forward this message to {server_uid}"
                )
            )

        client = None

        network_service = self.get_service(NetworkService)
        peer = network_service.stash.get_by_uid(self.verify_key, server_uid)

        if peer.is_ok() and peer.ok():
            peer = peer.ok()

            # Since we have several routes to a peer
            # we need to cache the client for a given server_uid along with the route
            peer_cache_key = hash(server_uid) + hash(peer.pick_highest_priority_route())
            if peer_cache_key in self.peer_client_cache:
                client = self.peer_client_cache[peer_cache_key]
            else:
                context = AuthedServiceContext(
                    server=self, credentials=api_call.credentials
                )

                client = peer.client_with_context(context=context)
                if client.is_err():
                    return SyftError(
                        message=f"Failed to create remote client for peer: "
                        f"{peer.id}. Error: {client.err()}"
                    )
                client = client.ok()

                self.peer_client_cache[peer_cache_key] = client

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

                if isinstance(result, BlobRetrievalByURL | SeaweedFSBlobDeposit):
                    result.proxy_server_uid = peer.id

            return result

        return SyftError(message=(f"Server has no route to {server_uid}"))

    def get_role_for_credentials(self, credentials: SyftVerifyKey) -> ServiceRole:
        role = self.get_service("userservice").get_role_for_credentials(
            credentials=credentials
        )
        return role

    def handle_api_call(
        self,
        api_call: SyftAPICall | SignedSyftAPICall,
        job_id: UID | None = None,
        check_call_location: bool = True,
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
        api_call: SyftAPICall | SignedSyftAPICall,
        job_id: UID | None = None,
        check_call_location: bool = True,
    ) -> Result | QueueItem | SyftObject | SyftError:
        if self.required_signed_calls and isinstance(api_call, SyftAPICall):
            return SyftError(
                message=f"You sent a {type(api_call)}. This server requires SignedSyftAPICall."
            )
        else:
            if not api_call.is_valid:
                return SyftError(message="Your message signature is invalid")

        if api_call.message.server_uid != self.id and check_call_location:
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
                server=self,
                credentials=credentials,
                role=role,
                job_id=job_id,
                is_blocking_api_call=is_blocking,
            )
            AuthServerContextRegistry.set_server_context(self.id, context, credentials)

            user_config_registry = UserServiceConfigRegistry.from_role(role)

            if api_call.path not in user_config_registry:
                if ServiceConfigRegistry.path_exists(api_call.path):
                    return SyftError(
                        message=f"As a `{role}`, "
                        f"you have no access to: {api_call.path}"
                    )
                else:
                    return SyftError(
                        message=f"API call not in registered services: {api_call.path}"
                    )

            _private_api_path = user_config_registry.private_path_for(api_call.path)
            method = self.get_service_method(_private_api_path)
            try:
                logger.info(f"API Call: {api_call}")
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

    def add_api_endpoint_execution_to_queue(
        self,
        credentials: SyftVerifyKey,
        method: str,
        path: str,
        *args: Any,
        worker_pool: str | None = None,
        **kwargs: Any,
    ) -> Job | SyftError:
        job_id = UID()
        task_uid = UID()
        worker_settings = WorkerSettings.from_server(server=self)

        if worker_pool is None:
            worker_pool = self.get_default_worker_pool()
        else:
            worker_pool = self.get_worker_pool_by_name(worker_pool)

        if isinstance(worker_pool, SyftError):
            return worker_pool
        elif worker_pool is None:
            return SyftError(message="Worker pool not found")

        # Create a Worker pool reference object
        worker_pool_ref = LinkedObject.from_obj(
            worker_pool,
            service_type=SyftWorkerPoolService,
            server_uid=self.id,
        )
        queue_item = APIEndpointQueueItem(
            id=task_uid,
            method=method,
            server_uid=self.id,
            syft_client_verify_key=credentials,
            syft_server_location=self.id,
            job_id=job_id,
            worker_settings=worker_settings,
            args=args,
            kwargs={"path": path, **kwargs},
            has_execute_permissions=True,
            worker_pool=worker_pool_ref,  # set worker pool reference as part of queue item
        )

        action = Action.from_api_endpoint_execution()
        return self.add_queueitem_to_queue(
            queue_item=queue_item,
            credentials=credentials,
            action=action,
            job_type=JobType.TWINAPIJOB,
        )

    def get_worker_pool_ref_by_name(
        self, credentials: SyftVerifyKey, worker_pool_name: str | None = None
    ) -> LinkedObject | SyftError:
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
            server_uid=self.id,
        )
        return worker_pool_ref

    def add_action_to_queue(
        self,
        action: Action,
        credentials: SyftVerifyKey,
        parent_job_id: UID | None = None,
        has_execute_permissions: bool = False,
        worker_pool_name: str | None = None,
    ) -> Job | SyftError:
        job_id = UID()
        task_uid = UID()
        worker_settings = WorkerSettings.from_server(server=self)

        # Extract worker pool id from user code
        if action.user_code_id is not None:
            result = self.user_code_stash.get_by_uid(
                credentials=credentials, uid=action.user_code_id
            )

            # If result is Ok, then user code object exists
            if result.is_ok() and result.ok() is not None:
                user_code = result.ok()
                worker_pool_name = user_code.worker_pool_name

        worker_pool_ref = self.get_worker_pool_ref_by_name(
            credentials, worker_pool_name
        )
        if isinstance(worker_pool_ref, SyftError):
            return worker_pool_ref
        queue_item = ActionQueueItem(
            id=task_uid,
            server_uid=self.id,
            syft_client_verify_key=credentials,
            syft_server_location=self.id,
            job_id=job_id,
            worker_settings=worker_settings,
            args=[],
            kwargs={"action": action},
            has_execute_permissions=has_execute_permissions,
            worker_pool=worker_pool_ref,  # set worker pool reference as part of queue item
        )
        user_id = self.get_service("UserService").get_user_id_for_credentials(
            credentials
        )

        return self.add_queueitem_to_queue(
            queue_item=queue_item,
            credentials=credentials,
            action=action,
            parent_job_id=parent_job_id,
            user_id=user_id,
        )

    def add_queueitem_to_queue(
        self,
        *,
        queue_item: QueueItem,
        credentials: SyftVerifyKey,
        action: Action | None = None,
        parent_job_id: UID | None = None,
        user_id: UID | None = None,
        job_type: JobType = JobType.JOB,
    ) -> Job | SyftError:
        log_id = UID()
        role = self.get_role_for_credentials(credentials=credentials)
        context = AuthedServiceContext(server=self, credentials=credentials, role=role)

        result_obj = ActionObject.empty()
        if action is not None:
            result_obj = ActionObject.obj_not_ready(id=action.result_id)
            result_obj.id = action.result_id
            result_obj.syft_resolved = False
            result_obj.syft_server_location = self.id
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
            server_uid=self.id,
            syft_client_verify_key=credentials,
            syft_server_location=self.id,
            log_id=log_id,
            parent_job_id=parent_job_id,
            action=action,
            requested_by=user_id,
            job_type=job_type,
        )

        # 🟡 TODO 36: Needs distributed lock
        job_res = self.job_stash.set(credentials, job)
        if job_res.is_err():
            return SyftError(message=f"{job_res.err()}")
        self.queue_stash.set_placeholder(credentials, queue_item)

        log_service = self.get_service("logservice")

        result = log_service.add(context, log_id, queue_item.job_id)
        if isinstance(result, SyftError):
            return result
        return job

    def _sort_jobs(self, jobs: list[Job]) -> list[Job]:
        job_datetimes = {}
        for job in jobs:
            try:
                d = datetime.strptime(job.creation_time, DATETIME_FORMAT)
            except Exception:
                d = datetime(MINYEAR, 1, 1)
            job_datetimes[job.id] = d

        jobs.sort(
            key=lambda job: (job.status != JobStatus.COMPLETED, job_datetimes[job.id]),
            reverse=True,
        )

        return jobs

    def _get_existing_user_code_jobs(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> list[Job] | SyftError:
        job_service = self.get_service("jobservice")
        jobs = job_service.get_by_user_code_id(
            context=context, user_code_id=user_code_id
        )

        if isinstance(jobs, SyftError):
            return jobs

        return self._sort_jobs(jobs)

    def _is_usercode_call_on_owned_kwargs(
        self,
        context: AuthedServiceContext,
        api_call: SyftAPICall,
        user_code_id: UID,
    ) -> bool:
        if api_call.path != "code.call":
            return False
        user_code_service = self.get_service("usercodeservice")
        return user_code_service.is_execution_on_owned_args(
            context, user_code_id, api_call.kwargs
        )

    def add_api_call_to_queue(
        self, api_call: SyftAPICall, parent_job_id: UID | None = None
    ) -> Job | SyftError:
        unsigned_call = api_call
        if isinstance(api_call, SignedSyftAPICall):
            unsigned_call = api_call.message

        credentials = api_call.credentials
        context = AuthedServiceContext(
            server=self,
            credentials=credentials,
            role=self.get_role_for_credentials(credentials=credentials),
        )

        is_user_code = unsigned_call.path == "code.call"

        service_str, method_str = unsigned_call.path.split(".")

        action = None
        if is_user_code:
            action = Action.from_api_call(unsigned_call)
            user_code_id = action.user_code_id

            user = self.get_service(UserService).get_current_user(context)
            if isinstance(user, SyftError):
                return user
            user = cast(UserView, user)

            is_execution_on_owned_kwargs_allowed = (
                user.mock_execution_permission or context.role == ServiceRole.ADMIN
            )
            is_usercode_call_on_owned_kwargs = self._is_usercode_call_on_owned_kwargs(
                context, unsigned_call, user_code_id
            )
            # Low side does not execute jobs, unless this is a mock execution
            if (
                not is_usercode_call_on_owned_kwargs
                and self.server_side_type == ServerSideType.LOW_SIDE
            ):
                existing_jobs = self._get_existing_user_code_jobs(context, user_code_id)
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

            elif (
                is_usercode_call_on_owned_kwargs
                and not is_execution_on_owned_kwargs_allowed
            ):
                return SyftError(
                    message="You do not have the permissions for mock execution, please contact the admin"
                )

            return self.add_action_to_queue(
                action, api_call.credentials, parent_job_id=parent_job_id
            )

        else:
            worker_settings = WorkerSettings.from_server(server=self)
            worker_pool_ref = self.get_worker_pool_ref_by_name(credentials=credentials)
            if isinstance(worker_pool_ref, SyftError):
                return worker_pool_ref

            queue_item = QueueItem(
                id=UID(),
                server_uid=self.id,
                syft_client_verify_key=api_call.credentials,
                syft_server_location=self.id,
                job_id=UID(),
                worker_settings=worker_settings,
                service=service_str,
                method=method_str,
                args=unsigned_call.args,
                kwargs=unsigned_call.kwargs,
                worker_pool=worker_pool_ref,
            )
            return self.add_queueitem_to_queue(
                queue_item=queue_item,
                credentials=api_call.credentials,
                action=None,
                parent_job_id=parent_job_id,
            )

    @property
    def pool_stash(self) -> SyftWorkerPoolStash:
        return self.get_service(SyftWorkerPoolService).stash

    @property
    def user_code_stash(self) -> UserCodeStash:
        return self.get_service(UserCodeService).stash

    def get_default_worker_pool(self) -> WorkerPool | None | SyftError:
        result = self.pool_stash.get_by_name(
            credentials=self.verify_key,
            pool_name=self.settings.default_worker_pool,
        )
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        worker_pool = result.ok()
        return worker_pool

    def get_worker_pool_by_name(self, name: str) -> WorkerPool | None | SyftError:
        result = self.pool_stash.get_by_name(
            credentials=self.verify_key, pool_name=name
        )
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        worker_pool = result.ok()
        return worker_pool

    def get_api(
        self,
        for_user: SyftVerifyKey | None = None,
        communication_protocol: PROTOCOL_TYPE | None = None,
    ) -> SyftAPI:
        return SyftAPI.for_user(
            server=self,
            user_verify_key=for_user,
            communication_protocol=communication_protocol,
        )

    def get_method_with_context(
        self, function: Callable, context: ServerServiceContext
    ) -> Callable:
        method = self.get_service_method(function)
        return partial(method, context=context)

    def get_unauthed_context(
        self, login_credentials: UserLoginCredentials
    ) -> ServerServiceContext:
        return UnauthedServiceContext(server=self, login_credentials=login_credentials)

    def create_initial_settings(self, admin_email: str) -> ServerSettings | None:
        try:
            settings_stash = SettingsStash(store=self.document_store)
            if self.signing_key is None:
                logger.debug(
                    "create_initial_settings failed as there is no signing key"
                )
                return None
            settings_exists = settings_stash.get_all(self.signing_key.verify_key).ok()
            if settings_exists:
                server_settings = settings_exists[0]
                if server_settings.__version__ != ServerSettings.__version__:
                    context = Context()
                    server_settings = server_settings.migrate_to(
                        ServerSettings.__version__, context
                    )
                    res = settings_stash.delete_by_uid(
                        self.signing_key.verify_key, server_settings.id
                    )
                    if res.is_err():
                        raise Exception(res.value)
                    res = settings_stash.set(
                        self.signing_key.verify_key, server_settings
                    )
                    if res.is_err():
                        raise Exception(res.value)
                self.name = server_settings.name
                self.association_request_auto_approval = (
                    server_settings.association_request_auto_approval
                )
                return None
            else:
                # Currently we allow automatic user registration on enclaves,
                # as enclaves do not have superusers
                if self.server_type == ServerType.ENCLAVE:
                    flags.CAN_REGISTER = True
                new_settings = ServerSettings(
                    id=self.id,
                    name=self.name,
                    verify_key=self.verify_key,
                    server_type=self.server_type,
                    deployed_on=datetime.now().date().strftime("%m/%d/%Y"),
                    signup_enabled=flags.CAN_REGISTER,
                    admin_email=admin_email,
                    server_side_type=self.server_side_type.value,  # type: ignore
                    show_warnings=self.enable_warnings,
                    association_request_auto_approval=self.association_request_auto_approval,
                    default_worker_pool=get_default_worker_pool_name(),
                )
                result = settings_stash.set(
                    credentials=self.signing_key.verify_key, settings=new_settings
                )
                if result.is_ok():
                    return result.ok()
                return None
        except Exception as e:
            logger.error("create_initial_settings failed", exc_info=e)
            return None


def create_admin_new(
    name: str,
    email: str,
    password: str,
    server: AbstractServer,
) -> User | None:
    try:
        user_stash = UserStash(store=server.document_store)
        row_exists = user_stash.get_by_email(
            credentials=server.signing_key.verify_key, email=email
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
            user.signing_key = server.signing_key
            user.verify_key = user.signing_key.verify_key
            result = user_stash.set(
                credentials=server.signing_key.verify_key,
                user=user,
                ignore_duplicates=True,
            )
            if result.is_ok():
                return result.ok()
            else:
                raise Exception(f"Could not create user: {result}")
    except Exception as e:
        logger.error("Unable to create new admin", exc_info=e)

    return None


class ServerRegistry:
    __server_registry__: dict[UID, Server] = {}

    @classmethod
    def set_server_for(
        cls,
        server_uid: UID | str,
        server: Server,
    ) -> None:
        if isinstance(server_uid, str):
            server_uid = UID.from_string(server_uid)

        cls.__server_registry__[server_uid] = server

    @classmethod
    def server_for(cls, server_uid: UID) -> Server:
        return cls.__server_registry__.get(server_uid, None)

    @classmethod
    def get_all_servers(cls) -> list[Server]:
        return list(cls.__server_registry__.values())

    @classmethod
    def remove_server(cls, server_uid: UID) -> None:
        if server_uid in cls.__server_registry__:
            del cls.__server_registry__[server_uid]


def get_default_worker_tag_by_env(dev_mode: bool = False) -> str | None:
    if in_kubernetes():
        return get_default_worker_image()
    elif dev_mode:
        return "local-dev"
    else:
        return __version__


def create_default_worker_pool(server: Server) -> SyftError | None:
    credentials = server.verify_key
    pull_image = not server.dev_mode
    image_stash = server.get_service(SyftWorkerImageService).stash
    default_pool_name = server.settings.default_worker_pool
    default_worker_pool = server.get_default_worker_pool()
    default_worker_tag = get_default_worker_tag_by_env(server.dev_mode)
    default_worker_pool_pod_annotations = get_default_worker_pool_pod_annotations()
    default_worker_pool_pod_labels = get_default_worker_pool_pod_labels()
    worker_count = get_default_worker_pool_count(server)
    context = AuthedServiceContext(
        server=server,
        credentials=credentials,
        role=ServiceRole.ADMIN,
    )

    if isinstance(default_worker_pool, SyftError):
        logger.error(
            f"Failed to get default worker pool {default_pool_name}. "
            f"Error: {default_worker_pool.message}"
        )
        return default_worker_pool

    logger.info(f"Creating default worker image with tag='{default_worker_tag}'. ")
    # Get/Create a default worker SyftWorkerImage
    default_image = create_default_image(
        credentials=credentials,
        image_stash=image_stash,
        tag=default_worker_tag,
        in_kubernetes=in_kubernetes(),
    )
    if isinstance(default_image, SyftError):
        logger.error(f"Failed to create default worker image: {default_image.message}")
        return default_image

    if not default_image.is_built:
        logger.info(f"Building default worker image with tag={default_worker_tag}. ")
        image_build_method = server.get_service_method(SyftWorkerImageService.build)
        # Build the Image for given tag
        result = image_build_method(
            context,
            image_uid=default_image.id,
            tag=DEFAULT_WORKER_IMAGE_TAG,
            pull_image=pull_image,
        )

        if isinstance(result, SyftError):
            logger.error(f"Failed to build default worker image: {result.message}")
            return None

    # Create worker pool if it doesn't exists
    logger.info(
        "Setting up worker pool"
        f"name={default_pool_name} "
        f"workers={worker_count} "
        f"image_uid={default_image.id} "
        f"in_memory={server.in_memory_workers}. "
    )
    if default_worker_pool is None:
        worker_to_add_ = worker_count
        create_pool_method = server.get_service_method(SyftWorkerPoolService.launch)
        result = create_pool_method(
            context,
            pool_name=default_pool_name,
            image_uid=default_image.id,
            num_workers=worker_count,
            pod_annotations=default_worker_pool_pod_annotations,
            pod_labels=default_worker_pool_pod_labels,
        )
    else:
        # Else add a worker to existing worker pool
        worker_to_add_ = max(default_worker_pool.max_count, worker_count) - len(
            default_worker_pool.worker_list
        )
        if worker_to_add_ > 0:
            add_worker_method = server.get_service_method(
                SyftWorkerPoolService.add_workers
            )
            result = add_worker_method(
                context=context,
                number=worker_to_add_,
                pool_name=default_pool_name,
            )
        else:
            return None

    if isinstance(result, SyftError):
        logger.info(f"Default worker pool error. {result.message}")
        return None

    for n in range(worker_to_add_):
        container_status = result[n]
        if container_status.error:
            logger.error(
                f"Failed to create container: Worker: {container_status.worker},"
                f"Error: {container_status.error}"
            )
            return None

    logger.info("Created default worker pool.")
    return None
