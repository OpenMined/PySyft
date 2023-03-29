# stdlib
from enum import Enum
from typing import Optional

# relative
from .credentials import SyftVerifyKey
from .serializable import serializable
from .uid import UID


@serializable(recursive_serde=True)
class ActionPermission(Enum):
    OWNER = 1
    READ = 2
    ALL_READ = 4
    WRITE = 8
    ALL_WRITE = 16
    EXECUTE = 32
    ALL_EXECUTE = 64


COMPOUND_ACTION_PERMISSION = set(
    [
        ActionPermission.ALL_READ,
        ActionPermission.ALL_WRITE,
        ActionPermission.ALL_EXECUTE,
    ]
)


@serializable(recursive_serde=True)
class ActionObjectPermission:
    def __init__(
        self,
        uid: UID,
        permission: ActionPermission,
        credentials: Optional[SyftVerifyKey] = None,
    ):
        if credentials is None:
            assert permission in COMPOUND_ACTION_PERMISSION
        self.uid = uid
        self.credentials = credentials
        self.permission = permission

    @property
    def permission_string(self) -> str:
        if self.permission in COMPOUND_ACTION_PERMISSION:
            return f"{self.permission.name}"
        else:
            return f"{self.credentials.verify}_{self.permission.name}"

    def __repr__(self) -> str:
        if self.credentials is not None:
            return f"<{self.permission.name}: {self.uid} as {self.credentials.verify}>"
        else:
            # TODO: somehow, __repr__ is only triggered in this case
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
