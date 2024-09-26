# stdlib

# third party

# third party
from sqlalchemy.orm import Session

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.db.stash import with_session
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from .worker_image import SyftWorkerImage


@serializable(canonical_name="SyftWorkerImageSQLStash", version=1)
class SyftWorkerImageStash(ObjectStash[SyftWorkerImage]):
    @as_result(SyftException, StashException, NotFoundException)
    @with_session
    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftWorkerImage,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
        session: Session = None,
        skip_check_type: bool = False,
    ) -> SyftWorkerImage:
        # By default syft images have all read permission
        add_permissions = [] if add_permissions is None else add_permissions
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
                session=session,
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
        # TODO cannot search on fields containing objects
        all_images = self.get_all(credentials=credentials).unwrap()
        for image in all_images:
            if image.config == config:
                return image
        raise NotFoundException(
            public_message=f"Worker Image with config {config} not found"
        )
