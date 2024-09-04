# stdlib
from enum import Enum
from typing import Any

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID


@serializable(canonical_name="ActionPermission", version=1)
class ActionPermission(Enum):
    OWNER = 1
    READ = 2
    ALL_READ = 4
    WRITE = 8
    ALL_WRITE = 32
    EXECUTE = 64
    ALL_EXECUTE = 128
    ALL_OWNER = 256

    @property
    def as_compound(self) -> "ActionPermission":
        if self in COMPOUND_ACTION_PERMISSION:
            return self
        elif self == ActionPermission.READ:
            return ActionPermission.ALL_READ
        elif self == ActionPermission.WRITE:
            return ActionPermission.ALL_WRITE
        elif self == ActionPermission.EXECUTE:
            return ActionPermission.ALL_EXECUTE
        elif self == ActionPermission.OWNER:
            return ActionPermission.ALL_OWNER
        else:
            raise Exception(f"Invalid compound permission {self}")


COMPOUND_ACTION_PERMISSION = {
    ActionPermission.ALL_READ,
    ActionPermission.ALL_WRITE,
    ActionPermission.ALL_EXECUTE,
    ActionPermission.ALL_OWNER,
}


@serializable(canonical_name="ActionObjectPermission", version=1)
class ActionObjectPermission:
    def __init__(
        self,
        uid: UID,
        permission: ActionPermission,
        credentials: SyftVerifyKey | None = None,
    ):
        if credentials is None:
            if permission not in COMPOUND_ACTION_PERMISSION:
                raise Exception(f"{permission} not in {COMPOUND_ACTION_PERMISSION}")
        self.uid = uid
        self.credentials = credentials
        self.permission = permission

    @classmethod
    def from_permission_string(
        cls, uid: UID, permission_string: str
    ) -> "ActionObjectPermission":
        if permission_string.startswith("ALL_"):
            permission = ActionPermission[permission_string]
            verify_key = None
        else:
            verify_key_str, perm_str = permission_string.split("_", 1)
            permission = ActionPermission[perm_str]
            verify_key = SyftVerifyKey.from_string(verify_key_str)

        return cls(uid=uid, permission=permission, credentials=verify_key)

    @property
    def permission_string(self) -> str:
        if self.permission in COMPOUND_ACTION_PERMISSION:
            return f"{self.permission.name}"
        else:
            if self.credentials is not None:
                return f"{self.credentials.verify}_{self.permission.name}"
            return f"{self.permission.name}"

    @property
    def compound_permission_string(self) -> str:
        return self.permission.as_compound.name

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "uid": str(self.uid),
            "credentials": str(self.credentials),
            "permission": str(self.permission),
        }

    def __repr__(self) -> str:
        if self.credentials is not None:
            return f"[{self.permission.name}: {self.uid} as {self.credentials.verify}]"
        else:
            # TODO: somehow, __repr__ is only triggered in this case.
            # Maybe fixed by change from <> brackets to [] so it now prints in htlm?
            return self.permission_string


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


@serializable(canonical_name="StoragePermission", version=1)
class StoragePermission:
    def __init__(self, uid: UID, server_uid: UID):
        self.uid = uid
        self.server_uid = server_uid

    def __repr__(self) -> str:
        return f"StoragePermission: {self.uid} on {self.server_uid}"

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "uid": str(self.uid),
            "server_uid": str(self.server_uid),
        }

    @property
    def permission_string(self) -> str:
        return str(self.server_uid)
