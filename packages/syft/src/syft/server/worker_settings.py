# future
from __future__ import annotations

# third party
from typing_extensions import Self

# relative
from ..abstract_server import AbstractServer
from ..abstract_server import ServerSideType
from ..abstract_server import ServerType
from ..deployment_type import DeploymentType
from ..serde.serializable import serializable
from ..server.credentials import SyftSigningKey
from ..service.queue.base_queue import QueueConfig
from ..store.blob_storage import BlobStorageConfig
from ..store.document_store import StoreConfig
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID


@serializable()
class WorkerSettings(SyftObject):
    __canonical_name__ = "WorkerSettings"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    server_type: ServerType
    server_side_type: ServerSideType
    deployment_type: DeploymentType = DeploymentType.REMOTE
    signing_key: SyftSigningKey
    document_store_config: StoreConfig
    action_store_config: StoreConfig
    blob_store_config: BlobStorageConfig | None = None
    queue_config: QueueConfig | None = None
    log_level: int | None = None

    @classmethod
    def from_server(cls, server: AbstractServer) -> Self:
        if server.server_side_type:
            server_side_type: str = server.server_side_type.value
        else:
            server_side_type = ServerSideType.HIGH_SIDE
        return cls(
            id=server.id,
            name=server.name,
            server_type=server.server_type,
            signing_key=server.signing_key,
            document_store_config=server.document_store_config,
            action_store_config=server.action_store_config,
            server_side_type=server_side_type,
            blob_store_config=server.blob_store_config,
            queue_config=server.queue_config,
            log_level=server.log_level,
            deployment_type=server.deployment_type,
        )
