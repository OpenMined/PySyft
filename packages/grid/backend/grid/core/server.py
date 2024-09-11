# syft absolute
from syft.abstract_server import ServerType
from syft.server.datasite import Datasite
from syft.server.datasite import Server
from syft.server.enclave import Enclave
from syft.server.gateway import Gateway
from syft.server.server import get_default_bucket_name
from syft.server.server import get_enable_warnings
from syft.server.server import get_server_name
from syft.server.server import get_server_side_type
from syft.server.server import get_server_type
from syft.server.server import get_server_uid_env
from syft.service.queue.zmq_client import ZMQClientConfig
from syft.service.queue.zmq_client import ZMQQueueConfig
from syft.store.blob_storage.seaweedfs import SeaweedFSClientConfig
from syft.store.blob_storage.seaweedfs import SeaweedFSConfig
from syft.store.mongo_client import MongoStoreClientConfig
from syft.store.mongo_document_store import MongoStoreConfig
from syft.store.sqlite_document_store import SQLiteStoreClientConfig
from syft.store.sqlite_document_store import SQLiteStoreConfig
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


def mongo_store_config() -> MongoStoreConfig:
    mongo_client_config = MongoStoreClientConfig(
        hostname=settings.MONGO_HOST,
        port=settings.MONGO_PORT,
        username=settings.MONGO_USERNAME,
        password=settings.MONGO_PASSWORD,
    )

    return MongoStoreConfig(client_config=mongo_client_config)


def sql_store_config() -> SQLiteStoreConfig:
    client_config = SQLiteStoreClientConfig(
        filename=f"{UID.from_string(get_server_uid_env())}.sqlite",
        path=settings.SQLITE_PATH,
    )
    return SQLiteStoreConfig(client_config=client_config)


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
store_config = sql_store_config() if single_container_mode else mongo_store_config()
blob_storage_config = None if single_container_mode else seaweedfs_config()
queue_config = queue_config()

worker: Server = worker_class(
    name=server_name,
    server_side_type=server_side_type,
    action_store_config=store_config,
    document_store_config=store_config,
    enable_warnings=enable_warnings,
    blob_storage_config=blob_storage_config,
    local_db=single_container_mode,
    queue_config=queue_config,
    migrate=True,
    in_memory_workers=settings.INMEMORY_WORKERS,
    smtp_username=settings.SMTP_USERNAME,
    smtp_password=settings.SMTP_PASSWORD,
    email_sender=settings.EMAIL_SENDER,
    smtp_port=settings.SMTP_PORT,
    smtp_host=settings.SMTP_HOST,
    association_request_auto_approval=settings.ASSOCIATION_REQUEST_AUTO_APPROVAL,
    background_tasks=True,
)
