# futureserver.py
# future
from __future__ import annotations

# stdlib
from collections import OrderedDict
from collections.abc import Callable
from datetime import MINYEAR
from datetime import datetime
from datetime import timezone
from functools import partial
import hashlib
import logging
import os
from pathlib import Path
import threading
from time import sleep
import traceback
from typing import Any
from typing import TypeVar
from typing import cast

# third party
from nacl.signing import SigningKey

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
from ..deployment_type import DeploymentType
from ..protocol.data_protocol import PROTOCOL_TYPE
from ..protocol.data_protocol import get_data_protocol
from ..service.action.action_object import Action
from ..service.action.action_object import ActionObject
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
from ..service.network.utils import PeerHealthCheckTask
from ..service.notifier.notifier_service import NotifierService
from ..service.output.output_service import OutputStash
from ..service.queue.base_queue import AbstractMessageHandler
from ..service.queue.base_queue import QueueConsumer
from ..service.queue.base_queue import QueueProducer
from ..service.queue.queue import APICallMessageHandler
from ..service.queue.queue import ConsumerType
from ..service.queue.queue import QueueManager
from ..service.queue.queue_stash import APIEndpointQueueItem
from ..service.queue.queue_stash import ActionQueueItem
from ..service.queue.queue_stash import QueueItem
from ..service.queue.queue_stash import QueueStash
from ..service.queue.zmq_client import QueueConfig
from ..service.queue.zmq_client import ZMQClientConfig
from ..service.queue.zmq_client import ZMQQueueConfig
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..service.service import AbstractService
from ..service.service import ServiceConfigRegistry
from ..service.service import UserServiceConfigRegistry
from ..service.settings.settings import ServerSettings
from ..service.settings.settings import ServerSettingsUpdate
from ..service.user.user import UserView
from ..service.user.user_roles import ServiceRole
from ..service.user.utils import create_root_admin_if_not_exists
from ..service.worker.utils import DEFAULT_WORKER_IMAGE_TAG
from ..service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ..service.worker.utils import create_default_image
from ..service.worker.worker_pool import WorkerPool
from ..service.worker.worker_pool_service import SyftWorkerPoolService
from ..service.worker.worker_pool_stash import SyftWorkerPoolStash
from ..service.worker.worker_stash import WorkerStash
from ..store.blob_storage import BlobStorageConfig
from ..store.blob_storage.on_disk import OnDiskBlobStorageClientConfig
from ..store.blob_storage.on_disk import OnDiskBlobStorageConfig
from ..store.blob_storage.seaweedfs import SeaweedFSBlobDeposit
from ..store.db.db import DBConfig
from ..store.db.db import DBManager
from ..store.db.postgres import PostgresDBConfig
from ..store.db.postgres import PostgresDBManager
from ..store.db.sqlite import SQLiteDBConfig
from ..store.db.sqlite import SQLiteDBManager
from ..store.db.stash import ObjectStash
from ..store.document_store_errors import NotFoundException
from ..store.document_store_errors import StashException
from ..store.linked_obj import LinkedObject
from ..types.datetime import DATETIME_FORMAT
from ..types.errors import SyftException
from ..types.result import Result
from ..types.result import as_result
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
from ..util.util import thread_ident
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .env import get_default_root_email
from .env import get_default_root_password
from .env import get_default_root_username
from .env import get_default_worker_image
from .env import get_default_worker_pool_name
from .env import get_default_worker_pool_pod_annotations
from .env import get_default_worker_pool_pod_labels
from .env import get_private_key_env
from .env import get_server_uid_env
from .env import get_syft_worker_uid
from .env import in_kubernetes
from .service_registry import ServiceRegistry
from .utils import get_named_server_uid
from .utils import get_temp_dir_for_server
from .utils import remove_temp_dir_for_server
from .worker_settings import WorkerSettings

logger = logging.getLogger(__name__)

SyftT = TypeVar("SyftT", bound=SyftObject)

# if user code needs to be serded and its not available we can call this to refresh
# the code for a specific server UID and thread
CODE_RELOADER: dict[int, Callable] = {}


def get_default_worker_pool_count(server: Server) -> int:
    return int(
        get_env(
            "DEFAULT_WORKER_POOL_COUNT", server.queue_config.client_config.n_consumers
        )
    )


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
        db_config: DBConfig | None = None,
        root_email: str | None = default_root_email,
        root_username: str | None = default_root_username,
        root_password: str | None = default_root_password,
        processes: int = 0,
        is_subprocess: bool = False,
        server_type: str | ServerType = ServerType.DATASITE,
        deployment_type: str | DeploymentType = "remote",
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
        log_level: int | None = None,
        smtp_username: str | None = None,
        smtp_password: str | None = None,
        email_sender: str | None = None,
        smtp_port: int | None = None,
        smtp_host: str | None = None,
        association_request_auto_approval: bool = False,
        background_tasks: bool = False,
        consumer_type: ConsumerType | None = None,
        db_url: str | None = None,
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
        self._settings = None

        if isinstance(server_type, str):
            server_type = ServerType(server_type)

        self.server_type = server_type

        if isinstance(deployment_type, str):
            deployment_type = DeploymentType(deployment_type)
        self.deployment_type = deployment_type

        # do this after we set the deployment type
        self.set_log_level(log_level)

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

        logger.critical(
            f"Hash of the signing key '{self.signing_key.deterministic_hash()[:5]}...'"
        )

        self.association_request_auto_approval = association_request_auto_approval

        consumer_type = (
            consumer_type or ConsumerType.Thread
            if thread_workers
            else ConsumerType.Process
        )
        self.queue_config = self.create_queue_config(
            n_consumers=n_consumers,
            create_producer=create_producer,
            consumer_type=consumer_type,
            queue_port=queue_port,
            queue_config=queue_config,
        )

        # must call before initializing stores
        if reset:
            self.remove_temp_dir()

        db_config = DBConfig.from_connection_string(db_url) if db_url else db_config

        if db_config is None:
            db_config = SQLiteDBConfig(
                filename=f"{self.id}_json.db",
                path=self.get_temp_dir("db"),
            )

        self.db_config = db_config

        self.db = self.init_stores(db_config=self.db_config)

        # construct services only after init stores
        self.services: ServiceRegistry = ServiceRegistry.for_server(self)
        self.db.init_tables(reset=reset)
        self.action_store = self.services.action.stash

        create_root_admin_if_not_exists(
            name=root_username,
            email=root_email,
            password=root_password,  # nosec
            server=self,
        )

        NotifierService.init_notifier(
            server=self,
            email_password=smtp_password,
            email_username=smtp_username,
            email_sender=email_sender,
            smtp_port=smtp_port,
            smtp_host=smtp_host,
        ).unwrap()

        self.post_init()

        if migrate:
            self.find_and_migrate_data()
        else:
            self.find_and_migrate_data([ServerSettings])

        self.create_initial_settings(admin_email=root_email).unwrap()

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
        if background_tasks:
            email_dispatcher = threading.Thread(
                target=self.email_notification_dispatcher, daemon=True
            )
            email_dispatcher.start()

    def email_notification_dispatcher(self) -> None:
        lock = threading.Lock()
        while True:
            # Use admin context to have access to the notifier obj
            context = AuthedServiceContext(
                server=self,
                credentials=self.verify_key,
                role=ServiceRole.ADMIN,
            )
            # Get notitifer settings
            notifier_settings = self.services.notifier.settings(
                context=context
            ).unwrap()
            lock.acquire()
            # Iterate over email_types and its queues
            # Ex: {'EmailRequest': {VerifyKey: [], VerifyKey: [], ...}}
            for email_template, email_queue in notifier_settings.email_queue.items():
                # Get the email frequency of that specific email type
                email_frequency = notifier_settings.email_frequency[email_template]
                for verify_key, queue in email_queue.items():
                    if self.services.notifier.is_time_to_dispatch(
                        email_frequency, datetime.now(timezone.utc)
                    ):
                        notifier_settings.send_batched_notification(
                            context=context, notification_queue=queue
                        ).unwrap()
                        notifier_settings.email_queue[email_template][verify_key] = []
                        self.services.notifier.stash.update(
                            credentials=self.verify_key, obj=notifier_settings
                        ).unwrap()
            lock.release()
            sleep(15)

    def set_log_level(self, log_level: int | str | None) -> None:
        def determine_log_level(
            log_level: str | int | None, default: int
        ) -> int | None:
            if log_level is None:
                return default
            if isinstance(log_level, str):
                level = logging.getLevelName(log_level.upper())
                if isinstance(level, str) and level.startswith("Level "):
                    level = logging.INFO  # defaults to info otherwise
                return level  # type: ignore
            return log_level

        default = logging.CRITICAL
        if self.deployment_type == DeploymentType.PYTHON:
            default = logging.CRITICAL
        elif self.dev_mode:  # if real deployment and dev mode
            default = logging.INFO

        self.log_level = determine_log_level(log_level, default)

        logging.getLogger().setLevel(self.log_level)

        if log_level == logging.DEBUG:
            # only do this if specifically set, very noisy
            logging.getLogger("uvicorn").setLevel(logging.DEBUG)
            logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
        else:
            logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
            logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)

    @property
    def runs_in_docker(self) -> bool:
        path = "/proc/self/cgroup"
        return (
            os.path.exists("/.dockerenv")
            or os.path.isfile(path)
            and any("docker" in line for line in open(path))
        )

    def init_blob_storage(self, config: BlobStorageConfig | None = None) -> None:
        if config is None:
            client_config = OnDiskBlobStorageClientConfig(
                base_directory=self.get_temp_dir("blob")
            )
            config_ = OnDiskBlobStorageConfig(
                client_config=client_config,
                min_blob_size=os.getenv("MIN_SIZE_BLOB_STORAGE_MB", 1),
            )
        else:
            config_ = config
        self.blob_store_config = config_
        self.blob_storage_client = config_.client_type(config=config_.client_config)

        # relative
        from ..store.blob_storage.seaweedfs import SeaweedFSConfig

        if isinstance(config, SeaweedFSConfig) and self.signing_key:
            remote_profiles = self.services.blob_storage.remote_profile_stash.get_all(
                credentials=self.signing_key.verify_key, has_permission=True
            ).unwrap()
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
        consumer_type: ConsumerType,
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
                consumer_type=consumer_type,
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

            if self.in_memory_workers:
                self.start_in_memory_workers(
                    address=address, message_handler=message_handler
                )

    def start_in_memory_workers(
        self, address: str, message_handler: type[AbstractMessageHandler]
    ) -> None:
        """Starts in-memory workers for the server."""

        worker_pools = self.pool_stash.get_all(credentials=self.verify_key).unwrap()
        for worker_pool in worker_pools:  # type: ignore
            # Skip the default worker pool
            if worker_pool.name == DEFAULT_WORKER_POOL_NAME:
                continue

            # Create consumers for each worker pool
            for linked_worker in worker_pool.worker_list:
                self.add_consumer_for_service(
                    service_name=worker_pool.name,
                    syft_worker_id=linked_worker.object_uid,
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
            worker_stash=self.worker_stash,  # type: ignore
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
        server_type: str | ServerType = ServerType.DATASITE,
        server_side_type: str | ServerSideType = ServerSideType.HIGH_SIDE,
        deployment_type: str | DeploymentType = "remote",
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
        consumer_type: ConsumerType | None = None,
        db_url: str | None = None,
        db_config: DBConfig | None = None,
        log_level: int | None = None,
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
            server_type=server_type,
            server_side_type=server_side_type,
            deployment_type=deployment_type,
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
            consumer_type=consumer_type,
            db_url=db_url,
            db_config=db_config,
            log_level=log_level,
        )

    def is_root(self, credentials: SyftVerifyKey) -> bool:
        return credentials == self.verify_key

    @property
    def root_client(self) -> SyftClient:
        # relative
        from ..client.client import PythonConnection

        connection = PythonConnection(server=self)
        client_type = connection.get_client_type().unwrap()
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

            try:
                migration_state = migration_state_service.get_state(
                    context, canonical_name
                ).unwrap()
                if migration_state.current_version != migration_state.latest_version:
                    klasses_to_be_migrated.append(object_type)
            except NotFoundException:
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
        return self.services.migration.migrate_data(
            context, document_store_object_types
        )

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

        client_type = connection.get_client_type().unwrap()

        guest_client = client_type(
            connection=connection, credentials=SyftSigningKey.generate()
        )
        if guest_client.api.refresh_api_callback is not None:
            guest_client.api.refresh_api_callback()
        return guest_client

    def __repr__(self) -> str:
        service_string = ""
        if not self.is_subprocess:
            services = [
                service.__class__.__name__ for service in self.initialized_services
            ]
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
            self.services.user_code.load_user_code(context=context)

        def reload_user_code() -> None:
            self.services.user_code.load_user_code(context=context)

        ti = thread_ident()
        if ti is not None:
            CODE_RELOADER[ti] = reload_user_code

    def init_stores(self, db_config: DBConfig) -> DBManager:
        if isinstance(db_config, SQLiteDBConfig):
            db = SQLiteDBManager(
                config=db_config,
                server_uid=self.id,
                root_verify_key=self.verify_key,
            )
        elif isinstance(db_config, PostgresDBConfig):
            db = PostgresDBManager(  # type: ignore
                config=db_config,
                server_uid=self.id,
                root_verify_key=self.verify_key,
            )
        else:
            raise SyftException(public_message=f"Unsupported DB config: {db_config}")

        self.queue_stash = QueueStash(store=db)

        print(f"Using {db_config.__class__.__name__} and {db_config.connection_string}")

        return db

    @property
    def job_stash(self) -> JobStash:
        return self.services.job.stash

    @property
    def output_stash(self) -> OutputStash:
        return self.services.output.stash

    @property
    def worker_stash(self) -> WorkerStash:
        return self.services.worker.stash

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

    @as_result(ValueError)
    def get_stash(self, object_type: SyftT) -> ObjectStash[SyftT]:
        if object_type not in self.services.stashes:
            raise ValueError(f"Stash for {object_type} not found.")
        return self.services.stashes[object_type]

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

    # NOTE: Some workflows currently expect the settings to be available,
    # even though they might not be defined yet. Because of this, we need to check
    # if the settings table is already defined. This function is basically a copy
    # of the settings property but ignoring stash error in case settings doesn't exist yet.
    # it should be removed once the settings are refactored and the inconsistencies between
    # settings and services are resolved.
    def get_settings(self) -> ServerSettings | None:
        if self._settings:
            return self._settings  # type: ignore
        if self.signing_key is None:
            raise ValueError(f"{self} has no signing key")

        settings_stash = self.services.settings.stash

        try:
            settings = settings_stash.get_all(self.signing_key.verify_key).unwrap()

            if len(settings) > 0:
                setting = settings[0]
                self.update_self(setting)
                self._settings = setting
                return setting
            else:
                return None

        except SyftException:
            return None

    @property
    def settings(self) -> ServerSettings:
        if self.signing_key is None:
            raise ValueError(f"{self} has no signing key")

        settings_stash = self.services.settings.stash
        error_msg = f"Cannot get server settings for '{self.name}'"

        all_settings = settings_stash.get_all(self.signing_key.verify_key).unwrap(
            public_message=error_msg
        )

        if len(all_settings) == 0:
            raise SyftException(public_message=error_msg)

        settings = all_settings[0]
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

    def await_future(self, credentials: SyftVerifyKey, uid: UID) -> QueueItem:
        # stdlib

        # relative
        from ..service.queue.queue import Status

        while True:
            result = self.queue_stash.pop_on_complete(credentials, uid).unwrap()
            if result.status == Status.COMPLETED:
                return result
            sleep(0.1)

    @instrument
    def resolve_future(self, credentials: SyftVerifyKey, uid: UID) -> QueueItem:
        queue_obj = self.queue_stash.pop_on_complete(credentials, uid).unwrap()
        queue_obj._set_obj_location_(
            server_uid=self.id,
            credentials=credentials,
        )
        return queue_obj

    @instrument
    def forward_message(
        self, api_call: SyftAPICall | SignedSyftAPICall
    ) -> Result | QueueItem | SyftObject | Any:
        server_uid = api_call.message.server_uid
        if "networkservice" not in self.service_path_map:
            raise SyftException(
                public_message=(
                    "Server has no network service so we can't "
                    f"forward this message to {server_uid}"
                )
            )

        client = None
        peer = self.services.network.stash.get_by_uid(
            self.verify_key, server_uid
        ).unwrap()

        # Since we have several routes to a peer
        # we need to cache the client for a given server_uid along with the route
        peer_cache_key = hash(server_uid) + hash(peer.pick_highest_priority_route())
        if peer_cache_key in self.peer_client_cache:
            client = self.peer_client_cache[peer_cache_key]
        else:
            context = AuthedServiceContext(
                server=self, credentials=api_call.credentials
            )

            client = peer.client_with_context(context=context).unwrap(
                public_message=f"Failed to create remote client for peer: {peer.id}"
            )
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
                result = debox_signed_syftapicall_response(
                    signed_result=signed_result
                ).unwrap()

                # relative
                from ..store.blob_storage import BlobRetrievalByURL

                if isinstance(result, BlobRetrievalByURL | SeaweedFSBlobDeposit):
                    result.proxy_server_uid = peer.id

            return result

        raise SyftException(public_message=(f"Server has no route to {server_uid}"))

    def get_role_for_credentials(self, credentials: SyftVerifyKey) -> ServiceRole:
        return self.services.user.get_role_for_credentials(
            credentials=credentials
        ).unwrap()

    @instrument
    def handle_api_call(
        self,
        api_call: SyftAPICall | SignedSyftAPICall,
        job_id: UID | None = None,
        check_call_location: bool = True,
    ) -> SignedSyftAPICall:
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
            raise SyftException(
                public_message=f"You sent a {type(api_call)}. This server requires SignedSyftAPICall."
            )
        else:
            if not api_call.is_valid:
                raise SyftException(public_message="Your message signature is invalid")

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

        credentials: SyftVerifyKey = api_call.credentials
        role = self.get_role_for_credentials(credentials=credentials)
        context = AuthedServiceContext(
            server=self,
            credentials=credentials,
            role=role,
            job_id=job_id,
            is_blocking_api_call=is_blocking,
        )

        if is_blocking or self.is_subprocess:
            api_call = api_call.message

            role = self.get_role_for_credentials(credentials=credentials)
            settings = self.get_settings()
            # TODO: This instance check should be removed once we can ensure that
            # self.settings will always return a ServerSettings object.
            if (
                settings is not None
                and isinstance(settings, ServerSettings)
                and not settings.allow_guest_sessions
                and role == ServiceRole.GUEST
            ):
                raise SyftException(
                    public_message="Server doesn't allow guest sessions."
                )
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
                    raise SyftException(
                        public_message=f"As a `{role}`, "
                        f"you have no access to: {api_call.path}"
                    )
                else:
                    raise SyftException(
                        public_message=f"API call not in registered services: {api_call.path}"
                    )

            _private_api_path = user_config_registry.private_path_for(api_call.path)
            method = self.get_service_method(_private_api_path)
            try:
                logger.info(f"API Call: {api_call}")

                result = method(context, *api_call.args, **api_call.kwargs)

                if isinstance(result, SyftError):
                    raise TypeError(
                        "Don't return a SyftError, raise SyftException instead"
                    )
                if not isinstance(result, SyftSuccess):
                    result = SyftSuccess(message="", value=result)
                result.add_warnings_from_context(context)
                tb = None
            except Exception as e:
                include_traceback = (
                    self.dev_mode or role.value >= ServiceRole.DATA_OWNER.value
                )
                result = SyftError.from_exception(
                    context=context, exc=e, include_traceback=include_traceback
                )
                if not include_traceback:
                    # then at least log it server side
                    if isinstance(e, SyftException):
                        tb = e.get_tb(context, overwrite_permission=True)
                    else:
                        tb = traceback.format_exc()
                    logger.debug(
                        f"Exception (hidden from DS) happened on the server side:\n{tb}"
                    )
        else:
            try:
                return self.add_api_call_to_queue(api_call)
            except SyftException as e:
                return SyftError.from_exception(context=context, exc=e)
            except Exception:
                result = SyftError(
                    message=f"Exception calling {api_call.path}. {traceback.format_exc()}"
                )
                tb = traceback.format_exc()
            if (
                isinstance(result, SyftError)
                and role.value < ServiceRole.DATA_OWNER.value
            ):
                print(f"Exception (hidden from DS) happened on the server side:\n{tb}")
        return result

    def add_api_endpoint_execution_to_queue(
        self,
        credentials: SyftVerifyKey,
        method: str,
        path: str,
        log_id: UID,
        *args: Any,
        worker_pool_name: str | None = None,
        **kwargs: Any,
    ) -> Job:
        job_id = UID()
        task_uid = UID()
        worker_settings = WorkerSettings.from_server(server=self)

        if worker_pool_name is None:
            worker_pool_name = self.get_default_worker_pool().unwrap()
        else:
            worker_pool_name = self.get_worker_pool_by_name(worker_pool_name).unwrap()

        # Create a Worker pool reference object
        worker_pool_ref = LinkedObject.from_obj(
            worker_pool_name,
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
            kwargs={"path": path, "log_id": log_id, **kwargs},
            has_execute_permissions=True,
            worker_pool=worker_pool_ref,  # set worker pool reference as part of queue item
        )

        action = Action.from_api_endpoint_execution()
        return self.add_queueitem_to_queue(
            queue_item=queue_item,
            credentials=credentials,
            action=action,
            job_type=JobType.TWINAPIJOB,
        ).unwrap()

    def get_worker_pool_ref_by_name(
        self, credentials: SyftVerifyKey, worker_pool_name: str | None = None
    ) -> LinkedObject:
        # If worker pool id is not set, then use default worker pool
        # Else, get the worker pool for given uid
        if worker_pool_name is None:
            worker_pool = self.get_default_worker_pool().unwrap()
        else:
            worker_pool = self.pool_stash.get_by_name(
                credentials, worker_pool_name
            ).unwrap()

        # Create a Worker pool reference object
        worker_pool_ref = LinkedObject.from_obj(
            worker_pool,
            service_type=SyftWorkerPoolService,
            server_uid=self.id,
        )
        return worker_pool_ref

    @instrument
    @as_result(SyftException)
    def add_action_to_queue(
        self,
        action: Action,
        credentials: SyftVerifyKey,
        parent_job_id: UID | None = None,
        has_execute_permissions: bool = False,
        worker_pool_name: str | None = None,
    ) -> Job:
        job_id = UID()
        task_uid = UID()
        worker_settings = WorkerSettings.from_server(server=self)

        # Extract worker pool id from user code
        if action.user_code_id is not None:
            user_code = self.user_code_stash.get_by_uid(
                credentials=credentials, uid=action.user_code_id
            ).unwrap()
            if user_code is not None:
                worker_pool_name = user_code.worker_pool_name

        worker_pool_ref = self.get_worker_pool_ref_by_name(
            credentials, worker_pool_name
        )
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
        user_id = self.services.user.get_user_id_for_credentials(credentials).unwrap()

        return self.add_queueitem_to_queue(
            queue_item=queue_item,
            credentials=credentials,
            action=action,
            parent_job_id=parent_job_id,
            user_id=user_id,
        ).unwrap()

    @instrument
    @as_result(SyftException)
    def add_queueitem_to_queue(
        self,
        *,
        queue_item: QueueItem,
        credentials: SyftVerifyKey,
        action: Action | None = None,
        parent_job_id: UID | None = None,
        user_id: UID | None = None,
        log_id: UID | None = None,
        job_type: JobType = JobType.JOB,
    ) -> Job:
        if log_id is None:
            log_id = UID()
        role = self.get_role_for_credentials(credentials=credentials)
        context = AuthedServiceContext(server=self, credentials=credentials, role=role)

        result_obj = ActionObject.empty()
        if action is not None:
            result_obj = ActionObject.obj_not_ready(
                id=action.result_id,
                syft_server_location=self.id,
                syft_client_verify_key=credentials,
            )
            result_obj.id = action.result_id
            result_obj.syft_resolved = False
            result_obj.syft_server_location = self.id
            result_obj.syft_client_verify_key = credentials

            if not self.services.action.stash.exists(
                credentials=credentials, uid=action.result_id
            ):
                self.services.action.set_result_to_store(
                    result_action_object=result_obj,
                    context=context,
                ).unwrap()

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
            endpoint=queue_item.kwargs.get("path", None),
        )

        # 🟡 TODO 36: Needs distributed lock
        self.job_stash.set(credentials, job).unwrap()
        self.queue_stash.set_placeholder(credentials, queue_item).unwrap()

        self.services.log.add(context, log_id, queue_item.job_id)

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

    @as_result(SyftException)
    def _get_existing_user_code_jobs(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> list[Job]:
        jobs = self.services.job.get_by_user_code_id(
            context=context, user_code_id=user_code_id
        )
        return self._sort_jobs(jobs)

    def _is_usercode_call_on_owned_kwargs(
        self,
        context: AuthedServiceContext,
        api_call: SyftAPICall,
        user_code_id: UID,
    ) -> bool:
        if api_call.path != "code.call":
            return False
        return self.services.user_code.is_execution_on_owned_args(
            context, user_code_id, api_call.kwargs
        )

    @instrument
    def add_api_call_to_queue(
        self, api_call: SyftAPICall, parent_job_id: UID | None = None
    ) -> SyftSuccess:
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

            user = self.services.user.get_current_user(context)
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
                try:
                    existing_jobs = self._get_existing_user_code_jobs(
                        context, user_code_id
                    ).unwrap()

                    if len(existing_jobs) > 0:
                        # relative
                        from ..util.util import prompt_warning_message

                        prompt_warning_message(
                            "There are existing jobs for this user code, returning the latest one"
                        )
                        return SyftSuccess(
                            message="Found multiple existing jobs, got last",
                            value=existing_jobs[-1],
                        )
                    else:
                        raise SyftException(
                            public_message="Please wait for the admin to allow the execution of this code"
                        )
                except Exception as e:
                    raise SyftException.from_exception(e)
            elif (
                is_usercode_call_on_owned_kwargs
                and not is_execution_on_owned_kwargs_allowed
            ):
                raise SyftException(
                    public_message="You do not have the permissions for mock execution, please contact the admin"
                )

            job = self.add_action_to_queue(
                action, api_call.credentials, parent_job_id=parent_job_id
            ).unwrap()

            return SyftSuccess(message="Succesfully queued job", value=job)

        else:
            worker_settings = WorkerSettings.from_server(server=self)
            worker_pool_ref = self.get_worker_pool_ref_by_name(credentials=credentials)
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
            ).unwrap()

    @property
    def pool_stash(self) -> SyftWorkerPoolStash:
        return self.services.syft_worker_pool.stash

    @property
    def user_code_stash(self) -> UserCodeStash:
        return self.services.user_code.stash

    @as_result(NotFoundException)
    def get_default_worker_pool(self) -> WorkerPool | None:
        return self.pool_stash.get_by_name(
            credentials=self.verify_key,
            pool_name=self.settings.default_worker_pool,
        ).unwrap()

    @as_result(NotFoundException)
    def get_worker_pool_by_name(self, name: str) -> WorkerPool:
        return self.pool_stash.get_by_name(
            credentials=self.verify_key, pool_name=name
        ).unwrap()

    @instrument
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

    @as_result(SyftException, StashException)
    def create_initial_settings(self, admin_email: str) -> ServerSettings:
        settings_stash = self.services.settings.stash

        if self.signing_key is None:
            logger.debug("create_initial_settings failed as there is no signing key")
            raise SyftException(
                public_message="create_initial_settings failed as there is no signing key"
            )

        settings_exists = settings_stash.get_all(self.signing_key.verify_key).unwrap()

        if settings_exists:
            server_settings = settings_exists[0]
            if server_settings.__version__ != ServerSettings.__version__:
                context = Context()
                server_settings = server_settings.migrate_to(
                    ServerSettings.__version__, context
                )
                settings_stash.delete_by_uid(
                    self.signing_key.verify_key, server_settings.id
                ).unwrap()
                settings_stash.set(
                    self.signing_key.verify_key, server_settings
                ).unwrap()
            self.name = server_settings.name
            self.association_request_auto_approval = (
                server_settings.association_request_auto_approval
            )
            return server_settings
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
                notifications_enabled=False,
            )

            return settings_stash.set(
                credentials=self.signing_key.verify_key, obj=new_settings
            ).unwrap()


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


def create_default_worker_pool(server: Server) -> None:
    credentials = server.verify_key
    pull_image = not server.dev_mode
    image_stash = server.services.syft_worker_image.stash
    default_pool_name = server.settings.default_worker_pool

    try:
        default_worker_pool = server.get_default_worker_pool().unwrap(
            public_message="Failed to get default worker pool"
        )
    except SyftException:
        default_worker_pool = None

    default_worker_tag = get_default_worker_tag_by_env(server.dev_mode)
    default_worker_pool_pod_annotations = get_default_worker_pool_pod_annotations()
    default_worker_pool_pod_labels = get_default_worker_pool_pod_labels()
    worker_count = get_default_worker_pool_count(server)
    context = AuthedServiceContext(
        server=server,
        credentials=credentials,
        role=ServiceRole.ADMIN,
    )

    logger.info(f"Creating default worker image with tag='{default_worker_tag}'. ")
    # Get/Create a default worker SyftWorkerImage
    # TODO: MERGE: Unwrap without public message?
    default_image = create_default_image(
        credentials=credentials,
        image_stash=image_stash,
        tag=default_worker_tag,
        in_kubernetes=in_kubernetes(),
    ).unwrap(public_message="Failed to create default worker image")

    if not default_image.is_built:
        logger.info(f"Building default worker image with tag={default_worker_tag}. ")
        # Build the Image for given tag
        result = server.services.worker_image.build(
            context,
            image_uid=default_image.id,
            tag=DEFAULT_WORKER_IMAGE_TAG,
            pull_image=pull_image,
        )

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
        result = server.services.syft_worker_pool.launch(
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
            result = server.services.syft_worker_pool.add_workers(
                context=context,
                number=worker_to_add_,
                pool_name=default_pool_name,
            )
        else:
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
