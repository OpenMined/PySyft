# stdlib
from enum import Enum
from typing import Any

# third party
from pydantic import model_validator

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftBaseObject
from ...types.uid import UID


@serializable()
class ActionPermission(Enum):
    OWNER = 1
    READ = 2
    ALL_READ = 4
    WRITE = 8
    ALL_WRITE = 32
    EXECUTE = 64
    ALL_EXECUTE = 128


COMPOUND_ACTION_PERMISSION = {
    ActionPermission.ALL_READ,
    ActionPermission.ALL_WRITE,
    ActionPermission.ALL_EXECUTE,
}


@serializable()
class ActionObjectPermission(SyftBaseObject):
    __canonical_name__ = "ActionObjectPermission"
    __version__ = SYFT_OBJECT_VERSION_1

    uid: UID
    credentials: SyftVerifyKey | None = None
    permission: ActionPermission

    @model_validator(mode="before")
    def validate_permissions(cls, values: dict) -> dict:
        if values.get("credentials") is None:
            permission = values.get("permission")
            if permission not in COMPOUND_ACTION_PERMISSION:
                raise ValueError(f"{permission} not in {COMPOUND_ACTION_PERMISSION}")
        return values

    @property
    def permission_string(self) -> str:
        if self.permission in COMPOUND_ACTION_PERMISSION:
            return f"{self.permission.name}"
        else:
            if self.credentials is not None:
                return f"{self.credentials.verify}_{self.permission.name}"
            return f"{self.permission.name}"

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


@serializable()
class StoragePermission(SyftBaseObject):
    __canonical_name__ = "StoragePermission"
    __version__ = SYFT_OBJECT_VERSION_1

    uid: UID
    node_uid: UID

    def __repr__(self) -> str:
        return f"StoragePermission: {self.uid} on {self.node_uid}"

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "uid": str(self.uid),
            "node_uid": str(self.node_uid),
        }
