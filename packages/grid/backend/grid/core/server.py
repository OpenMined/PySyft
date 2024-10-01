# stdlib
from pathlib import Path

# syft absolute
from syft.abstract_server import ServerType
from syft.server.datasite import Datasite
from syft.server.datasite import Server
from syft.server.enclave import Enclave
from syft.server.env import get_default_bucket_name
from syft.server.env import get_enable_warnings
from syft.server.env import get_server_name
from syft.server.env import get_server_side_type
from syft.server.env import get_server_type
from syft.server.env import get_server_uid_env
from syft.server.gateway import Gateway
from syft.service.queue.zmq_client import ZMQClientConfig
from syft.service.queue.zmq_client import ZMQQueueConfig
from syft.store.blob_storage.seaweedfs import SeaweedFSClientConfig
from syft.store.blob_storage.seaweedfs import SeaweedFSConfig
from syft.store.db.postgres import PostgresDBConfig
from syft.store.db.sqlite import SQLiteDBConfig
from syft.types.uid import UID

# server absolute
from grid.core.config import settings


def queue_config() -> ZMQQueueConfig:
    queue_config = ZMQQueueConfig(
        client_config=ZMQClientConfig(
            create_producer=settings.CREATE_PRODUCER,
            queue_port=settings.QUEUE_PORT,
            n_consumers=settings.N_CONSUMERS,
            consumer_service=settings.CONSUMER_SERVICE_NAME,
        )
    )
    return queue_config


def sql_store_config() -> SQLiteDBConfig:
    # Check if the directory exists, and create it if it doesn't
    sqlite_path = Path(settings.SQLITE_PATH)
    if not sqlite_path.exists():
        sqlite_path.mkdir(parents=True, exist_ok=True)

    return SQLiteDBConfig(
        filename=f"{UID.from_string(get_server_uid_env())}.sqlite",
        path=settings.SQLITE_PATH,
    )


def postgresql_store_config() -> PostgresDBConfig:
    return PostgresDBConfig(
        host=settings.POSTGRESQL_HOST,
        port=settings.POSTGRESQL_PORT,
        user=settings.POSTGRESQL_USERNAME,
        password=settings.POSTGRESQL_PASSWORD,
        database=settings.POSTGRESQL_DBNAME,
    )


def seaweedfs_config() -> SeaweedFSConfig:
    seaweed_client_config = SeaweedFSClientConfig(
        host=settings.S3_ENDPOINT,
        port=settings.S3_PORT,
        access_key=settings.S3_ROOT_USER,
        secret_key=settings.S3_ROOT_PWD,
        region=settings.S3_REGION,
        default_bucket_name=get_default_bucket_name(),
        mount_port=settings.SEAWEED_MOUNT_PORT,
    )

    return SeaweedFSConfig(
        client_config=seaweed_client_config,
        min_blob_size=settings.MIN_SIZE_BLOB_STORAGE_MB,
    )


server_type = ServerType(get_server_type())
server_name = get_server_name()

server_side_type = get_server_side_type()
enable_warnings = get_enable_warnings()

worker_classes = {
    ServerType.DATASITE: Datasite,
    ServerType.GATEWAY: Gateway,
    ServerType.ENCLAVE: Enclave,
}

worker_class = worker_classes[server_type]

single_container_mode = settings.SINGLE_CONTAINER_MODE
db_config = sql_store_config() if single_container_mode else postgresql_store_config()

blob_storage_config = None if single_container_mode else seaweedfs_config()
queue_config = queue_config()

worker: Server = worker_class(
    name=server_name,
    server_side_type=server_side_type,
    enable_warnings=enable_warnings,
    blob_storage_config=blob_storage_config,
    queue_config=queue_config,
    migrate=False,
    in_memory_workers=settings.INMEMORY_WORKERS,
    smtp_username=settings.SMTP_USERNAME,
    smtp_password=settings.SMTP_PASSWORD,
    email_sender=settings.EMAIL_SENDER,
    smtp_port=settings.SMTP_PORT,
    smtp_host=settings.SMTP_HOST,
    association_request_auto_approval=settings.ASSOCIATION_REQUEST_AUTO_APPROVAL,
    background_tasks=True,
    db_config=db_config,
)
