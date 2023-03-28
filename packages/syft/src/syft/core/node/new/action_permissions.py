# stdlib
from enum import Enum

# relative
from .credentials import SyftVerifyKey
from .serializable import serializable
from .uid import UID


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
