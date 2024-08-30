# stdlib

# third party

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from .worker_image import SyftWorkerImage

WorkerConfigPK = PartitionKey(key="config", type_=WorkerConfig)


@serializable(canonical_name="SyftWorkerImageStash", version=1)
class SyftWorkerImageStash(NewBaseUIDStoreStash):
    object_type = SyftWorkerImage
    settings: PartitionSettings = PartitionSettings(
        name=SyftWorkerImage.__canonical_name__,
        object_type=SyftWorkerImage,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(SyftException, StashException, NotFoundException)
    def set(  # type: ignore
        self,
        credentials: SyftVerifyKey,
        obj: SyftWorkerImage,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> SyftWorkerImage:
        add_permissions = [] if add_permissions is None else add_permissions

        # By default syft images have all read permission
        add_permissions.append(
            ActionObjectPermission(uid=obj.id, permission=ActionPermission.ALL_READ)
        )

        if isinstance(obj.config, DockerWorkerConfig):
            worker_config_exists = self.worker_config_exists(
                credentials=credentials, config=obj.config
            ).unwrap()
            if worker_config_exists:
                raise SyftException(
                    public_message=f"Worker Image with config {obj.config} already exists"
                )

        return (
            super()
            .set(
                credentials,
                obj,
                add_permissions=add_permissions,
                add_storage_permission=add_storage_permission,
                ignore_duplicates=ignore_duplicates,
            )
            .unwrap()
        )

    @as_result(StashException, NotFoundException)
    def worker_config_exists(
        self, credentials: SyftVerifyKey, config: WorkerConfig
    ) -> bool:
        try:
            self.get_by_worker_config(credentials=credentials, config=config).unwrap()
            return True
        except NotFoundException:
            return False

    @as_result(StashException, NotFoundException)
    def get_by_worker_config(
        self, credentials: SyftVerifyKey, config: WorkerConfig
    ) -> SyftWorkerImage:
        qks = QueryKeys(qks=[WorkerConfigPK.with_obj(config)])
        return self.query_one(credentials=credentials, qks=qks).unwrap(
            public_message=f"Worker Image with config {config} not found"
        )
