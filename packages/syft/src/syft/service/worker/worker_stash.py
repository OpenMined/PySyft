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
from .worker_pool import ConsumerState
from .worker_pool import SyftWorker


@serializable(canonical_name="WorkerSQLStash", version=1)
class WorkerStash(ObjectStash[SyftWorker]):
    @as_result(StashException)
    @with_session
    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftWorker,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
        session: Session = None,
        skip_check_type: bool = False,
    ) -> SyftWorker:
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
                ignore_duplicates=ignore_duplicates,
                add_storage_permission=add_storage_permission,
                session=session,
            )
            .unwrap()
        )

    @as_result(StashException, NotFoundException)
    def update_consumer_state(
        self, credentials: SyftVerifyKey, worker_uid: UID, consumer_state: ConsumerState
    ) -> SyftWorker:
        worker = self.get_by_uid(credentials=credentials, uid=worker_uid).unwrap()
        worker.consumer_state = consumer_state
        return self.update(credentials=credentials, obj=worker).unwrap()
