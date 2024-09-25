# stdlib

# third party

# third party
from sqlalchemy.orm import Session

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.db.stash import with_session
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from .worker_pool import WorkerPool


@serializable(canonical_name="SyftWorkerPoolSQLStash", version=1)
class SyftWorkerPoolStash(ObjectStash[WorkerPool]):
    @as_result(StashException, NotFoundException)
    def get_by_name(self, credentials: SyftVerifyKey, pool_name: str) -> WorkerPool:
        result = self.get_one(
            credentials=credentials,
            filters={"name": pool_name},
        )

        return result.unwrap(
            public_message=f"WorkerPool with name {pool_name} not found"
        )

    @as_result(StashException)
    @with_session
    def set(
        self,
        credentials: SyftVerifyKey,
        obj: WorkerPool,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
        session: Session = None,
        skip_check_type: bool = False,
    ) -> WorkerPool:
        # By default all worker pools have all read permission
        add_permissions = [] if add_permissions is None else add_permissions
        add_permissions.append(
            ActionObjectPermission(uid=obj.id, permission=ActionPermission.ALL_READ)
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

    @as_result(StashException)
    def get_by_image_uid(
        self, credentials: SyftVerifyKey, image_uid: UID
    ) -> list[WorkerPool]:
        return self.get_all(
            credentials=credentials,
            filters={"image_id": image_uid},
        ).unwrap()
