# syft absolute
from syft.abstract_node import NodeType
from syft.node.domain import Domain
from syft.node.enclave import Enclave
from syft.node.gateway import Gateway
from syft.node.node import get_enable_warnings
from syft.node.node import get_node_name
from syft.node.node import get_node_side_type
from syft.node.node import get_node_type
from syft.node.node import get_node_uid_env
from syft.service.queue.zmq_queue import ZMQClientConfig
from syft.service.queue.zmq_queue import ZMQQueueConfig
from syft.store.blob_storage.seaweedfs import SeaweedFSClientConfig
from syft.store.blob_storage.seaweedfs import SeaweedFSConfig
from syft.store.mongo_client import MongoStoreClientConfig
from syft.store.mongo_document_store import MongoStoreConfig
from syft.store.sqlite_document_store import SQLiteStoreClientConfig
from syft.store.sqlite_document_store import SQLiteStoreConfig

# grid absolute
from grid.core.config import settings


def queue_config() -> ZMQQueueConfig:
    queue_config = ZMQQueueConfig(
        client_config=ZMQClientConfig(
            create_producer=settings.CREATE_PRODUCER,
            queue_port=settings.QUEUE_PORT,
            n_consumers=settings.N_CONSUMERS,
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
    client_config = SQLiteStoreClientConfig(path=settings.SQLITE_PATH)
    return SQLiteStoreConfig(client_config=client_config)


def seaweedfs_config() -> SeaweedFSConfig:
    seaweed_client_config = SeaweedFSClientConfig(
        host=settings.S3_ENDPOINT,
        port=settings.S3_PORT,
        access_key=settings.S3_ROOT_USER,
        secret_key=settings.S3_ROOT_PWD,
        region=settings.S3_REGION,
        default_bucket_name=get_node_uid_env(),
        mount_port=settings.SEAWEED_MOUNT_PORT,
    )

    return SeaweedFSConfig(client_config=seaweed_client_config)


node_type = NodeType(get_node_type())
node_name = get_node_name()

node_side_type = get_node_side_type()
enable_warnings = get_enable_warnings()

worker_classes = {
    NodeType.DOMAIN: Domain,
    NodeType.GATEWAY: Gateway,
    NodeType.ENCLAVE: Enclave,
}

worker_class = worker_classes[node_type]

single_container_mode = settings.SINGLE_CONTAINER_MODE
store_config = sql_store_config() if single_container_mode else mongo_store_config()
blob_storage_config = None if single_container_mode else seaweedfs_config()
queue_config = queue_config()

worker = worker_class(
    name=node_name,
    node_side_type=node_side_type,
    action_store_config=store_config,
    document_store_config=store_config,
    enable_warnings=enable_warnings,
    blob_storage_config=blob_storage_config,
    local_db=single_container_mode,
    queue_config=queue_config,
    migrate=True,
)
