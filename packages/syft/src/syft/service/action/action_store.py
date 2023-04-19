# future
from __future__ import annotations

# stdlib
from typing import List
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.dict_document_store import DictStoreConfig
from ...store.document_store import BasePartitionSettings
from ...store.document_store import StoreConfig
from ...types.syft_object import SyftObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..response import SyftSuccess
from .action_permissions import ActionObjectEXECUTE
from .action_permissions import ActionObjectOWNER
from .action_permissions import ActionObjectPermission
from .action_permissions import ActionObjectREAD
from .action_permissions import ActionObjectWRITE
from .action_permissions import ActionPermission


class ActionStore:
    pass


@serializable()
class KeyValueActionStore(ActionStore):
    """Generic Key-Value Action store.

    Parameters:
        store_config: StoreConfig
            Backend specific configuration, including connection configuration, database name, or client class type.
        root_verify_key: Optional[SyftVerifyKey]
            Signature verification key, used for checking access permissions.
    """

    def __init__(
        self, store_config: StoreConfig, root_verify_key: Optional[SyftVerifyKey] = None
    ) -> None:
        self.store_config = store_config
        self.settings = BasePartitionSettings(name="Action")
        self.data = self.store_config.backing_store(
            "data", self.settings, self.store_config
        )
        self.permissions = self.store_config.backing_store(
            "permissions", self.settings, self.store_config, ddtype=set
        )
        if root_verify_key is None:
            root_verify_key = SyftSigningKey.generate().verify_key
        self.root_verify_key = root_verify_key

    def get(self, uid: UID, credentials: SyftVerifyKey) -> Result[SyftObject, str]:
        # TODO 🟣 Temporarily added skip permission argument for enclave
        # until permissions are fully integrated
        # if you get something you need READ permission
        read_permission = ActionObjectREAD(uid=uid, credentials=credentials)
        # if True:
        if self.has_permission(read_permission):
            syft_object = self.data[uid]
            return Ok(syft_object)
        return Err(f"Permission: {read_permission} denied")

    def get_pointer(
        self, uid: UID, credentials: SyftVerifyKey, node_uid: UID
    ) -> Result[SyftObject, str]:
        try:
            # 🟡 TODO 34: do we want pointer read permissions?
            if uid in self.data:
                obj = self.data[uid]
                if isinstance(obj, TwinObject):
                    obj = obj.mock
                obj.syft_point_to(node_uid)
                return Ok(obj)
            return Err("Permission denied")
        except Exception as e:
            return Err(str(e))

    def exists(self, uid: UID) -> bool:
        return uid in self.data

    def set(
        self, uid: UID, credentials: SyftVerifyKey, syft_object: SyftObject
    ) -> Result[SyftSuccess, Err]:
        # if you set something you need WRITE permission
        write_permission = ActionObjectWRITE(uid=uid, credentials=credentials)
        can_write = self.has_permission(write_permission)

        if not self.exists(uid=uid):
            # attempt to claim it for writing
            ownership_result = self.take_ownership(uid=uid, credentials=credentials)
            can_write = True if ownership_result.is_ok() else False

        if can_write:
            self.data[uid] = syft_object
            if uid not in self.permissions:
                # create default permissions
                self.permissions[uid] = set()
            permission = f"{credentials.verify}_READ"
            permissions = self.permissions[uid]
            permissions.add(permission)
            self.permissions[uid] = permissions
            return Ok(SyftSuccess(message=f"Set for ID: {uid}"))
        return Err(f"Permission: {write_permission} denied")

    def take_ownership(
        self, uid: UID, credentials: SyftVerifyKey
    ) -> Result[SyftSuccess, str]:
        # first person using this UID can claim ownership
        if uid not in self.permissions and uid not in self.data:
            self.add_permissions(
                [
                    ActionObjectOWNER(uid=uid, credentials=credentials),
                    ActionObjectWRITE(uid=uid, credentials=credentials),
                    ActionObjectREAD(uid=uid, credentials=credentials),
                    ActionObjectEXECUTE(uid=uid, credentials=credentials),
                ]
            )
            return Ok(SyftSuccess(message=f"Ownership of ID: {uid} taken."))
        return Err(f"UID: {uid} already owned.")

    def delete(self, uid: UID, credentials: SyftVerifyKey) -> Result[SyftSuccess, str]:
        # if you delete something you need OWNER permission
        # is it bad to evict a key and have someone else reuse it?
        # perhaps we should keep permissions but no data?
        owner_permission = ActionObjectOWNER(uid=uid, credentials=credentials)
        if self.has_permission(owner_permission):
            if uid in self.data:
                del self.data[uid]
            if uid in self.permissions:
                del self.permissions[uid]
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return Err(f"Permission: {owner_permission} denied")

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        if not isinstance(permission.permission, ActionPermission):
            raise Exception(f"ObjectPermission type: {permission.permission} not valid")

        if self.root_verify_key.verify == permission.credentials.verify:
            return True

        if (
            permission.uid in self.permissions
            and permission.permission_string in self.permissions[permission.uid]
        ):
            return True

        # 🟡 TODO 14: add ALL_READ, ALL_EXECUTE etc
        if permission.permission == ActionPermission.OWNER:
            pass
        elif permission.permission == ActionPermission.READ:
            pass
        elif permission.permission == ActionPermission.WRITE:
            pass
        elif permission.permission == ActionPermission.EXECUTE:
            pass

        return False

    def add_permission(self, permission: ActionObjectPermission) -> None:
        permissions = self.permissions[permission.uid]
        permissions.add(permission.permission_string)
        self.permissions[permission.uid] = permissions

    def remove_permission(self, permission: ActionObjectPermission):
        permissions = self.permissions[permission.uid]
        permissions.remove(permission.permission_string)
        self.permissions[permission.uid] = permissions

    def add_permissions(self, permissions: List[ActionObjectPermission]) -> None:
        results = []
        for permission in permissions:
            results.append(self.add_permission(permission))


@serializable()
class DictActionStore(KeyValueActionStore):
    """Dictionary-Based Key-Value Action store.

    Parameters:
        store_config: StoreConfig
            Backend specific configuration, including client class type.
        root_verify_key: Optional[SyftVerifyKey]
            Signature verification key, used for checking access permissions.
    """

    def __init__(
        self,
        store_config: Optional[StoreConfig] = None,
        root_verify_key: Optional[SyftVerifyKey] = None,
    ) -> None:
        store_config = store_config if store_config is not None else DictStoreConfig()
        super().__init__(store_config=store_config, root_verify_key=root_verify_key)


@serializable()
class SQLiteActionStore(KeyValueActionStore):
    """SQLite-Based Key-Value Action store.

    Parameters:
        store_config: StoreConfig
            SQLite specific configuration, including connection settings or client class type.
        root_verify_key: Optional[SyftVerifyKey]
            Signature verification key, used for checking access permissions.
    """

    pass
