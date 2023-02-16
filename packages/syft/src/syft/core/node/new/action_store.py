# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from enum import Enum
from typing import List
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .response import SyftSuccess


@serializable(recursive_serde=True)
class ActionPermission(Enum):
    OWNER = 1
    READ = 2
    WRITE = 4
    EXECUTE = 8


@serializable(recursive_serde=True)
class ActionObjectPermission:
    def __init__(
        self, uid: UID, credentials: SyftVerifyKey, permission: ActionPermission
    ):
        self.uid = uid
        self.credentials = credentials
        self.permission = permission

    @property
    def permission_string(self) -> str:
        return f"{self.credentials.verify}_{self.permission.name}"

    def __repr__(self) -> str:
        return f"<{self.permission.name}: {self.uid} as {self.credentials.verify}>"


class ActionObjectOWNER(ActionObjectPermission):
    def __init__(self, uid: UID, credentials: SyftVerifyKey):
        self.uid = uid
        self.credentials = credentials
        self.permission = ActionPermission.OWNER


class ActionObjectREAD(ActionObjectPermission):
    def __init__(self, uid: UID, credentials: SyftVerifyKey):
        self.uid = uid
        self.credentials = credentials
        self.permission = ActionPermission.READ


class ActionObjectWRITE(ActionObjectPermission):
    def __init__(self, uid: UID, credentials: SyftVerifyKey):
        self.uid = uid
        self.credentials = credentials
        self.permission = ActionPermission.WRITE


class ActionObjectEXECUTE(ActionObjectPermission):
    def __init__(self, uid: UID, credentials: SyftVerifyKey):
        self.uid = uid
        self.credentials = credentials
        self.permission = ActionPermission.EXECUTE


@serializable(recursive_serde=True)
class ActionStore:
    def __init__(self, root_verify_key: Optional[SyftVerifyKey] = None) -> None:
        self.data = {}
        self.permissions = defaultdict(set)
        if root_verify_key is None:
            root_verify_key = SyftSigningKey.generate().verify_key
        self.root_verify_key = root_verify_key

    def get(
        self, uid: UID, credentials: SyftVerifyKey, skip_permission: bool = False
    ) -> Result[SyftObject, str]:
        # if you get something you need READ permission
        read_permission = ActionObjectREAD(uid=uid, credentials=credentials)
        if skip_permission or self.has_permission(read_permission):
            data = self.data[uid]
            syft_object = SyftObject.from_mongo(data)
            return Ok(syft_object)
        return Err(f"Permission: {read_permission} denied")

    def get_pointer(
        self, uid: UID, credentials: SyftVerifyKey, node_uid: UID
    ) -> Result[SyftObject, str]:
        # 🟡 TODO 34: do we want pointer read permissions?
        if uid in self.data:
            data = self.data[uid]
            syft_object_ptr = SyftObject.from_mongo(data).to_pointer(node_uid)
            if syft_object_ptr:
                return Ok(syft_object_ptr)
        return Err("Permission denied")

    def exists(self, uid: UID) -> bool:
        return uid in self.data

    def set(
        self, uid: UID, credentials: SyftVerifyKey, syft_object: SyftObject
    ) -> Result[SyftSuccess, Err]:
        # if you set something you need WRITE permission
        write_permission = ActionObjectWRITE(uid=uid, credentials=credentials)
        can_write = self.has_permission(write_permission)

        # if the user cant write maybe it doesnt exist
        if not can_write and not self.exists(uid=uid):
            # attempt to claim it for writing
            ownership_result = self.take_ownership(uid=uid, credentials=credentials)
            can_write = True if ownership_result.is_ok() else False

        if can_write:
            self.data[
                uid
            ] = (
                syft_object.to_mongo()
            )  # 🟡 TODO 13: Create to_storage interface with Mongo first
            if uid not in self.permissions:
                # create default permissions
                self.permissions[uid] = set()
            permission = f"{credentials.verify}_READ"
            self.permissions[uid].add(permission)
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
            del self.data[uid]
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
        self.permissions[permission.uid].add(permission.permission_string)

    def remove_permission(self, permission: ActionObjectPermission):
        self.permissions[permission.uid].remove(permission.permission_string)

    def add_permissions(self, permissions: List[ActionObjectPermission]) -> None:
        results = []
        for permission in permissions:
            results.append(self.add_permission(permission))
