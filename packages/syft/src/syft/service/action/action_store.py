# future
from __future__ import annotations

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...types.syft_object import SyftObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..response import SyftSuccess
from .action_object import ActionObject
from .action_object import is_action_data_empty
from .action_permissions import ActionObjectEXECUTE
from .action_permissions import ActionObjectREAD
from .action_permissions import ActionObjectWRITE
from .action_permissions import StoragePermission


@serializable(canonical_name="ActionObjectSQLStore", version=1)
class ActionObjectStash(ObjectStash[ActionObject]):
    object_type = ActionObject

    def get(
        self, uid: UID, credentials: SyftVerifyKey, has_permission: bool = False
    ) -> Result[ActionObject, str]:
        uid = uid.id  # We only need the UID from LineageID or UID

        # TODO remove and use get_by_uid instead
        result_or_err = self.get_by_uid(
            credentials=credentials,
            uid=uid,
            has_permission=has_permission,
        )
        if result_or_err.is_err():
            return Err(result_or_err.err())

        result = result_or_err.ok()
        if result is None:
            return Err(f"Could not find item with uid {uid}")
        return Ok(result)

    def get_mock(self, credentials: SyftVerifyKey, uid: UID) -> Result[SyftObject, str]:
        uid = uid.id  # We only need the UID from LineageID or UID

        try:
            obj_or_err = self.get_by_uid(
                credentials=credentials, uid=uid, has_permission=True
            )  # type: ignore
            if obj_or_err.is_err():
                return Err(obj_or_err.err())
            obj = obj_or_err.ok()
            if isinstance(obj, TwinObject) and not is_action_data_empty(obj.mock):
                return Ok(obj.mock)
            return Err("No mock")
        except Exception as e:
            return Err(f"Could not find item with uid {uid}, {e}")

    def get_pointer(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        server_uid: UID,
    ) -> Result[SyftObject, str]:
        uid = uid.id  # We only need the UID from LineageID or UID

        try:
            result_or_err = self.get_by_uid(
                credentials=credentials, uid=uid, has_permission=True
            )
            has_permissions = self.has_permission(
                ActionObjectREAD(uid=uid, credentials=credentials)
            )
            if result_or_err.is_err():
                return Err(result_or_err.err())

            obj = result_or_err.ok()
            if obj is None:
                return Err("Permission denied")

            if has_permissions:
                if isinstance(obj, TwinObject):
                    return Ok(obj.private.syft_point_to(server_uid))
                return Ok(obj.syft_point_to(server_uid))

            # if its a twin with a mock anyone can have this
            if isinstance(obj, TwinObject):
                return Ok(obj.mock.syft_point_to(server_uid))

            # finally worst case you get ActionDataEmpty so you can still trace
            return Ok(obj.as_empty().syft_point_to(server_uid))

        except Exception as e:
            return Err(str(e))

    def set_or_update(  # type: ignore
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        syft_object: SyftObject,
        has_result_read_permission: bool = False,
        add_storage_permission: bool = True,
    ) -> Result[SyftSuccess, Err]:
        uid = uid.id  # We only need the UID from LineageID or UID

        if self.exists(credentials=credentials, uid=uid):
            permissions = []
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
            )
            self.add_permissions(permissions)
            self.add_storage_permissions(storage_permission)
            return Ok(SyftSuccess(message=f"Set for ID: {uid}"))

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
        )

        return Ok(SyftSuccess(message=f"Set for ID: {uid}"))

    def set(self, *args, **kwargs):  # type: ignore
        raise Exception("Use `ActionObjectStash.set_or_update` instead.")
