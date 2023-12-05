# future
from __future__ import annotations

# stdlib
import threading
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
from ...types.uid import LineageID
from ...types.uid import UID
from ..response import SyftSuccess
from .action_object import is_action_data_empty
from .action_permissions import ActionObjectEXECUTE
from .action_permissions import ActionObjectOWNER
from .action_permissions import ActionObjectPermission
from .action_permissions import ActionObjectREAD
from .action_permissions import ActionObjectWRITE
from .action_permissions import ActionPermission

lock = threading.RLock()


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

    def get(
        self, uid: UID, credentials: SyftVerifyKey, has_permission=False
    ) -> Result[SyftObject, str]:
        uid = uid.id  # We only need the UID from LineageID or UID

        # if you get something you need READ permission
        read_permission = ActionObjectREAD(uid=uid, credentials=credentials)
        if has_permission or self.has_permission(read_permission):
            try:
                if isinstance(uid, LineageID):
                    syft_object = self.data[uid.id]
                elif isinstance(uid, UID):
                    syft_object = self.data[uid]
                else:
                    raise Exception(f"Unrecognized UID type: {type(uid)}")
                return Ok(syft_object)
            except Exception as e:
                return Err(f"Could not find item with uid {uid}, {e}")
        return Err(f"Permission: {read_permission} denied")

    def get_mock(self, uid: UID) -> Result[SyftObject, str]:
        uid = uid.id  # We only need the UID from LineageID or UID

        try:
            syft_object = self.data[uid]
            if isinstance(syft_object, TwinObject) and not is_action_data_empty(
                syft_object.mock
            ):
                return Ok(syft_object.mock)
            return Err("No mock")
        except Exception as e:
            return Err(f"Could not find item with uid {uid}, {e}")

    def get_pointer(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        node_uid: UID,
    ) -> Result[SyftObject, str]:
        uid = uid.id  # We only need the UID from LineageID or UID

        try:
            if uid in self.data:
                obj = self.data[uid]
                read_permission = ActionObjectREAD(uid=uid, credentials=credentials)

                # if you have permission you can have private data
                if self.has_permission(read_permission):
                    if isinstance(obj, TwinObject):
                        return Ok(obj.private.syft_point_to(node_uid))
                    return Ok(obj.syft_point_to(node_uid))

                # if its a twin with a mock anyone can have this
                if isinstance(obj, TwinObject):
                    return Ok(obj.mock.syft_point_to(node_uid))

                # finally worst case you get ActionDataEmpty so you can still trace
                return Ok(obj.as_empty().syft_point_to(node_uid))

            return Err("Permission denied")
        except Exception as e:
            return Err(str(e))

    def exists(self, uid: UID) -> bool:
        uid = uid.id  # We only need the UID from LineageID or UID

        return uid in self.data

    def set(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        syft_object: SyftObject,
        has_result_read_permission: bool = False,
    ) -> Result[SyftSuccess, Err]:
        uid = uid.id  # We only need the UID from LineageID or UID

        # if you set something you need WRITE permission
        write_permission = ActionObjectWRITE(uid=uid, credentials=credentials)
        can_write = self.has_permission(write_permission)

        if not self.exists(uid=uid):
            # attempt to claim it for writing
            if has_result_read_permission:
                ownership_result = self.take_ownership(uid=uid, credentials=credentials)
                can_write = True if ownership_result.is_ok() else False
            else:
                # root takes owneship, but you can still write
                ownership_result = self.take_ownership(
                    uid=uid, credentials=self.root_verify_key
                )
                can_write = True if ownership_result.is_ok() else False

        if can_write:
            self.data[uid] = syft_object
            if has_result_read_permission:
                if uid not in self.permissions:
                    # create default permissions
                    self.permissions[uid] = set()
                self.add_permission(ActionObjectREAD(uid=uid, credentials=credentials))
            else:
                self.add_permissions(
                    [
                        ActionObjectWRITE(uid=uid, credentials=credentials),
                        ActionObjectEXECUTE(uid=uid, credentials=credentials),
                    ]
                )

            return Ok(SyftSuccess(message=f"Set for ID: {uid}"))
        return Err(f"Permission: {write_permission} denied")

    def take_ownership(
        self, uid: UID, credentials: SyftVerifyKey
    ) -> Result[SyftSuccess, str]:
        uid = uid.id  # We only need the UID from LineageID or UID

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
        uid = uid.id  # We only need the UID from LineageID or UID

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

    def has_permissions(self, permissions: List[ActionObjectPermission]) -> bool:
        return all(self.has_permission(p) for p in permissions)

    def add_permission(self, permission: ActionObjectPermission) -> None:
        permissions = self.permissions[permission.uid]
        permissions.add(permission.permission_string)
        self.permissions[permission.uid] = permissions

    def remove_permission(self, permission: ActionObjectPermission):
        permissions = self.permissions[permission.uid]
        permissions.remove(permission.permission_string)
        self.permissions[permission.uid] = permissions

    def add_permissions(self, permissions: List[ActionObjectPermission]) -> None:
        for permission in permissions:
            self.add_permission(permission)

    def migrate_data(self, to_klass: SyftObject, credentials: SyftVerifyKey):
        has_root_permission = credentials == self.root_verify_key

        if has_root_permission:
            for key, value in self.data.items():
                try:
                    if value.__canonical_name__ != to_klass.__canonical_name__:
                        continue
                    migrated_value = value.migrate_to(to_klass.__version__)
                except Exception as e:
                    return Err(
                        f"Failed to migrate data to {to_klass} for qk: {key}. Exception: {e}"
                    )
                result = self.set(
                    uid=key,
                    credentials=credentials,
                    syft_object=migrated_value,
                )

                if result.is_err():
                    return result.err()

            return Ok(True)

        return Err("You don't have permissions to migrate data.")


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


@serializable()
class MongoActionStore(KeyValueActionStore):
    """Mongo-Based  Action store.

    Parameters:
        store_config: StoreConfig
            Mongo specific configuration.
        root_verify_key: Optional[SyftVerifyKey]
            Signature verification key, used for checking access permissions.
    """

    pass
