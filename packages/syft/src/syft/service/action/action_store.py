# future
from __future__ import annotations

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import SyftObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from .action_object import ActionObject
from .action_object import is_action_data_empty
from .action_permissions import ActionObjectEXECUTE
from .action_permissions import ActionObjectPermission
from .action_permissions import ActionObjectREAD
from .action_permissions import ActionObjectWRITE
from .action_permissions import StoragePermission


@serializable(canonical_name="ActionObjectSQLStore", version=1)
class ActionObjectStash(ObjectStash[ActionObject]):
    # We are storing ActionObject, Action, TwinObject
    allow_any_type = True

    @as_result(NotFoundException, SyftException)
    def get(
        self, uid: UID, credentials: SyftVerifyKey, has_permission: bool = False
    ) -> ActionObject:
        uid = uid.id  # We only need the UID from LineageID or UID
        # TODO remove and use get_by_uid instead
        return self.get_by_uid(
            credentials=credentials,
            uid=uid,
            has_permission=has_permission,
        ).unwrap()

    @as_result(NotFoundException, SyftException)
    def get_mock(self, credentials: SyftVerifyKey, uid: UID) -> SyftObject:
        uid = uid.id  # We only need the UID from LineageID or UID

        obj = self.get_by_uid(
            credentials=credentials, uid=uid, has_permission=True
        ).unwrap()
        if isinstance(obj, TwinObject) and not is_action_data_empty(obj.mock):
            return obj.mock
        raise NotFoundException(public_message=f"No mock found for object {uid}")

    @as_result(NotFoundException, SyftException)
    def get_pointer(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        server_uid: UID,
    ) -> SyftObject:
        uid = uid.id  # We only need the UID from LineageID or UID

        obj = self.get_by_uid(
            credentials=credentials, uid=uid, has_permission=True
        ).unwrap()
        has_permissions = self.has_permission(
            ActionObjectREAD(uid=uid, credentials=credentials)
        )

        if has_permissions:
            if isinstance(obj, TwinObject):
                return obj.private.syft_point_to(server_uid)
            return obj.syft_point_to(server_uid)  # type: ignore

        # if its a twin with a mock anyone can have this
        if isinstance(obj, TwinObject):
            return obj.mock.syft_point_to(server_uid)

        # finally worst case you get ActionDataEmpty so you can still trace
        return obj.as_empty().syft_point_to(server_uid)  # type: ignore

    @as_result(SyftException, StashException)
    def set_or_update(  # type: ignore
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        syft_object: SyftObject,
        has_result_read_permission: bool = False,
        add_storage_permission: bool = True,
    ) -> UID:
        uid = uid.id  # We only need the UID from LineageID or UID

        if self.exists(credentials=credentials, uid=uid):
            permissions: list[ActionObjectPermission] = []
            if has_result_read_permission:
                permissions.append(ActionObjectREAD(uid=uid, credentials=credentials))
            else:
                permissions.extend(
                    [
                        ActionObjectWRITE(uid=uid, credentials=credentials),
                        ActionObjectEXECUTE(uid=uid, credentials=credentials),
                    ]
                )
            storage_permission = []
            if add_storage_permission:
                storage_permission.append(
                    StoragePermission(uid=uid, server_uid=self.server_uid)
                )

            self.update(
                credentials=credentials,
                obj=syft_object,
            ).unwrap()
            self.add_permissions(permissions).unwrap()
            self.add_storage_permissions(storage_permission).unwrap()
            return uid

        owner_credentials = (
            credentials if has_result_read_permission else self.root_verify_key
        )
        # if not has_result_read_permission
        # root takes owneship, but you can still write and execute
        super().set(
            credentials=owner_credentials,
            obj=syft_object,
            add_permissions=[
                ActionObjectWRITE(uid=uid, credentials=credentials),
                ActionObjectEXECUTE(uid=uid, credentials=credentials),
            ],
            add_storage_permission=add_storage_permission,
        ).unwrap()

        return uid

    def set(self, *args, **kwargs):  # type: ignore
        raise Exception("Use `ActionObjectStash.set_or_update` instead.")
