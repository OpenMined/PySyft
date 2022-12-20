# future
from __future__ import annotations

# stdlib
from enum import Enum
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

# third party
from bcrypt import gensalt
from bcrypt import hashpw
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ....core.node.common.node_table.syft_object import transform
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .credentials import SyftCredentials
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .transforms import drop
from .transforms import keep
from .transforms import make_set_default


class ServiceRoleCapability(Enum):
    CAN_MAKE_DATA_REQUESTS = 1
    CAN_TRIAGE_DATA_REQUESTS = 2
    CAN_MANAGE_PRIVACY_BUDGET = 4
    CAN_CREATE_USERS = 8
    CAN_MANAGE_USERS = 16
    CAN_EDIT_ROLES = 32
    CAN_MANAGE_INFRASTRUCTURE = 64
    CAN_UPLOAD_DATA = 128
    CAN_UPLOAD_LEGAL_DOCUMENT = 256
    CAN_EDIT_DOMAIN_SETTINGS = 512


@serializable(recursive_serde=True)
class ServiceRole(Enum):
    GUEST = 1


class User(SyftObject):
    # version
    __canonical_name__ = "User"
    __version__ = 1

    # fields
    email: str
    name: str
    hashed_password: str
    salt: str
    signing_key: SyftSigningKey
    verify_key: SyftVerifyKey
    role: ServiceRole
    created_at: Optional[str]

    # serde / storage rules
    __attr_state__ = [
        "email",
        "name",
        "hashed_password",
        "salt",
        "signing_key",
        "verify_key",
        "role",
        "created_at",
    ]
    __attr_searchable__ = ["name", "email", "verify_key"]
    __attr_unique__ = ["email", "signing_key"]


def default_role(role: ServiceRole) -> Callable:
    return make_set_default(key="role", value=role)


def hash_password(output: dict):
    if output["password"] == output["password_verify"]:
        salt, hashed = __salt_and_hash_password(output["password"], 12)
        output["hashed_password"] = hashed
        output["salt"] = salt
    return output


def generate_key(output: dict) -> dict:
    signing_key = SyftSigningKey.generate()
    output["signing_key"] = signing_key
    output["verify_key"] = signing_key.verify_key
    return output


def __salt_and_hash_password(password: str, rounds: int) -> Tuple[str, str]:
    bytes_pass = password.encode("UTF-8")
    salt = gensalt(rounds=rounds)
    salt_len = len(salt)
    hashed = hashpw(bytes_pass, salt)
    hashed = hashed[salt_len:]
    hashed = hashed.decode("UTF-8")
    salt = salt.decode("UTF-8")
    return salt, hashed


class UserUpdate(SyftObject):
    __canonical_name__ = "UserUpdate"
    __version__ = 1

    email: str
    name: str
    role: Optional[ServiceRole] = None  # make sure role cant be set without uid
    password: Optional[str] = None
    password_verify: Optional[str] = None


@transform(UserUpdate, User)
def user_update_to_user() -> List[Callable]:
    return [
        hash_password,
        generate_key,
        default_role(ServiceRole.GUEST),
        drop(["password", "password_verify"]),
    ]


@transform(User, UserUpdate)
def user_to_update_user() -> List[Callable]:
    return [keep(["id", "email", "name", "role"])]


class UserCollection:
    def __init__(self) -> None:
        self.data = {}
        self.primary_keys = {}

    def create(
        self, credentials: SyftCredentials, user_update: UserUpdate
    ) -> Result[UserUpdate, str]:
        if user_update.id is None:
            user_update.id = UID()
        user = user_update.to(User)

        result = self.set(uid=user.id, credentials=credentials, syft_object=user)
        if result.is_ok():
            return Ok(user.to(UserUpdate))
        else:
            return Err("Failed to create User.")

    def view(self, uid: UID, credentials: SyftCredentials) -> Result[UserUpdate, str]:
        user_result = self.get(uid=uid, credentials=credentials)
        if user_result.is_ok():
            return Ok(user_result.ok().to(UserUpdate))
        else:
            return Err(f"Failed to get User for UID: {uid}")

    def set(
        self, uid: UID, credentials: SyftCredentials, syft_object: SyftObject
    ) -> Result[bool, str]:
        self.data[uid] = syft_object.to_mongo()
        return Ok(True)

    def get(self, uid: UID, credentials: SyftCredentials) -> Result[SyftObject, str]:
        if uid not in self.data:
            return Err(f"UID: {uid} not in {type(self)} store.")
        syft_object = SyftObject.from_mongo(self.data[uid])
        return Ok(syft_object)
