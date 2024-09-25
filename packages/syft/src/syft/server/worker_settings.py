# future
from __future__ import annotations

# stdlib
from collections.abc import Callable

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
from ..store.db.db import DBConfig
from ..store.document_store import StoreConfig
from ..types.syft_migration import migrate
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SYFT_OBJECT_VERSION_2
from ..types.syft_object import SyftObject
from ..types.transforms import TransformContext
from ..types.transforms import drop
from ..types.uid import UID


@serializable()
class WorkerSettings(SyftObject):
    __canonical_name__ = "WorkerSettings"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    name: str
    server_type: ServerType
    server_side_type: ServerSideType
    deployment_type: DeploymentType = DeploymentType.REMOTE
    signing_key: SyftSigningKey
    db_config: DBConfig
    blob_store_config: BlobStorageConfig | None = None
    queue_config: QueueConfig | None = None
    log_level: int | None = None

    @classmethod
    def from_server(cls, server: AbstractServer) -> Self:
        server_side_type = server.server_side_type or ServerSideType.HIGH_SIDE
        return cls(
            id=server.id,
            name=server.name,
            server_type=server.server_type,
            signing_key=server.signing_key,
            db_config=server.db_config,
            server_side_type=server_side_type,
            blob_store_config=server.blob_store_config,
            queue_config=server.queue_config,
            log_level=server.log_level,
            deployment_type=server.deployment_type,
        )


@serializable()
class WorkerSettingsV1(SyftObject):
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


def set_db_config(context: TransformContext) -> TransformContext:
    if context.output:
        context.output["db_config"] = (
            context.server.db_config if context.server is not None else DBConfig()
        )
    return context


@migrate(WorkerSettingsV1, WorkerSettings)
def migrate_workersettings_v1_to_v2() -> list[Callable]:
    return [
        drop("document_store_config"),
        drop("action_store_config"),
        set_db_config,
    ]
