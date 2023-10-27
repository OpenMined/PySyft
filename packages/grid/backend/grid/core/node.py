# syft absolute
from syft.node.domain import Domain
from syft.node.gateway import Gateway
from syft.node.node import get_enable_warnings
from syft.node.node import get_node_name
from syft.node.node import get_node_side_type
from syft.node.node import get_node_type
from syft.node.node import get_node_uid_env
from syft.store.blob_storage.seaweedfs import SeaweedFSClientConfig
from syft.store.blob_storage.seaweedfs import SeaweedFSConfig
from syft.store.mongo_client import MongoStoreClientConfig
from syft.store.mongo_document_store import MongoStoreConfig
from syft.store.sqlite_document_store import SQLiteStoreClientConfig
from syft.store.sqlite_document_store import SQLiteStoreConfig

# grid absolute
from grid.core.config import settings

mongo_client_config = MongoStoreClientConfig(
    hostname=settings.MONGO_HOST,
    port=settings.MONGO_PORT,
    username=settings.MONGO_USERNAME,
    password=settings.MONGO_PASSWORD,
)

mongo_store_config = MongoStoreConfig(client_config=mongo_client_config)


client_config = SQLiteStoreClientConfig(path=settings.SQLITE_PATH)
sql_store_config = SQLiteStoreConfig(client_config=client_config)

node_type = get_node_type()
node_name = get_node_name()

node_side_type = get_node_side_type()
enable_warnings = get_enable_warnings()

seaweed_client_config = SeaweedFSClientConfig(
    host=settings.S3_ENDPOINT,
    port=settings.S3_PORT,
    access_key=settings.S3_ROOT_USER,
    secret_key=settings.S3_ROOT_PWD,
    region=settings.S3_REGION,
    bucket_name=get_node_uid_env(),
)

blob_storage_config = SeaweedFSConfig(client_config=seaweed_client_config)


if node_type == "gateway" or node_type == "network":
    worker = Gateway(
        name=node_name,
        node_side_type=node_side_type,
        action_store_config=mongo_store_config,
        document_store_config=mongo_store_config,
        enable_warnings=enable_warnings,
        blob_storage_config=blob_storage_config,
    )
else:
    worker = Domain(
        name=node_name,
        node_side_type=node_side_type,
        action_store_config=mongo_store_config,
        document_store_config=mongo_store_config,
        enable_warnings=enable_warnings,
        blob_storage_config=blob_storage_config,
    )
