# stdlib

# third party
from result import Err, Result

# relative
from ...custom_worker.config import DockerWorkerConfig, WorkerConfig
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import (
    BaseUIDStoreStash,
    DocumentStore,
    PartitionKey,
    PartitionSettings,
    QueryKeys,
)
from ..action.action_permissions import ActionObjectPermission, ActionPermission
from .worker_image import SyftWorkerImage

WorkerConfigPK = PartitionKey(key="config", type_=WorkerConfig)


@serializable(canonical_name="SyftWorkerImageStash", version=1)
class SyftWorkerImageStash(BaseUIDStoreStash):
    object_type = SyftWorkerImage
    settings: PartitionSettings = PartitionSettings(
        name=SyftWorkerImage.__canonical_name__,
        object_type=SyftWorkerImage,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftWorkerImage,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[SyftWorkerImage, str]:
        add_permissions = [] if add_permissions is None else add_permissions

        # By default syft images have all read permission
        add_permissions.append(
            ActionObjectPermission(uid=obj.id, permission=ActionPermission.ALL_READ),
        )

        if isinstance(obj.config, DockerWorkerConfig):
            result = self.get_by_worker_config(
                credentials=credentials, config=obj.config,
            )
            if result.is_ok() and result.ok() is not None:
                return Err(f"Image already exists for: {obj.config}")

        return super().set(
            credentials,
            obj,
            add_permissions=add_permissions,
            add_storage_permission=add_storage_permission,
            ignore_duplicates=ignore_duplicates,
        )

    def get_by_worker_config(
        self, credentials: SyftVerifyKey, config: WorkerConfig,
    ) -> Result[SyftWorkerImage | None, str]:
        qks = QueryKeys(qks=[WorkerConfigPK.with_obj(config)])
        return self.query_one(credentials=credentials, qks=qks)
