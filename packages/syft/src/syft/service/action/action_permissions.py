# stdlib
from enum import Enum
from typing import Optional

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
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
class ActionObjectPermission:
    def __init__(
        self,
        uid: UID,
        permission: ActionPermission,
        credentials: Optional[SyftVerifyKey] = None,
    ):
        if credentials is None:
            if permission not in COMPOUND_ACTION_PERMISSION:
                raise Exception(f"{permission} not in {COMPOUND_ACTION_PERMISSION}")
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
