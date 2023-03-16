# stdlib
from enum import Enum
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

# third party
from bcrypt import checkpw
from bcrypt import gensalt
from bcrypt import hashpw
import pydantic
from pydantic.networks import EmailStr

# relative
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .transforms import TransformContext
from .transforms import drop
from .transforms import generate_id
from .transforms import keep
from .transforms import make_set_default
from .transforms import transform
from .transforms import validate_email
from .uid import UID


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
    ADMIN = 0
    GUEST = 1


@serializable(recursive_serde=True)
class User(SyftObject):
    # version
    __canonical_name__ = "User"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]

    @pydantic.validator("email", pre=True, always=True)
    def make_email(cls, v: EmailStr) -> EmailStr:
        return EmailStr(v)

    # fields
    email: Optional[EmailStr]
    name: Optional[str]
    hashed_password: Optional[str]
    salt: Optional[str]
    signing_key: Optional[SyftSigningKey]
    verify_key: Optional[SyftVerifyKey]
    role: Optional[ServiceRole]
    institution: Optional[str]
    website: Optional[str] = None
    created_at: Optional[str]

    # serde / storage rules
    __attr_state__ = [
        "id",
        "email",
        "name",
        "hashed_password",
        "salt",
        "signing_key",
        "verify_key",
        "role",
        "institution",
        "website",
        "created_at",
    ]
    __attr_searchable__ = ["name", "email", "verify_key", "role"]
    __attr_unique__ = ["email", "signing_key", "verify_key"]
    __attr_repr_cols__ = ["name", "email"]


def default_role(role: ServiceRole) -> Callable:
    return make_set_default(key="role", value=role)


def hash_password(context: TransformContext) -> TransformContext:
    if context.output["password"] is not None and (
        context.output["password"] == context.output["password_verify"]
    ):
        salt, hashed = salt_and_hash_password(context.output["password"], 12)
        context.output["hashed_password"] = hashed
        context.output["salt"] = salt
    return context


def generate_key(context: TransformContext) -> TransformContext:
    signing_key = SyftSigningKey.generate()
    context.output["signing_key"] = signing_key
    context.output["verify_key"] = signing_key.verify_key
    return context


def salt_and_hash_password(password: str, rounds: int) -> Tuple[str, str]:
    bytes_pass = password.encode("UTF-8")
    salt = gensalt(rounds=rounds)
    hashed = hashpw(bytes_pass, salt)
    hashed_bytes = hashed.decode("UTF-8")
    salt_bytes = salt.decode("UTF-8")
    return salt_bytes, hashed_bytes


def check_pwd(password: str, hashed_password: str) -> bool:
    return checkpw(
        password=password.encode("utf-8"),
        hashed_password=hashed_password.encode("utf-8"),
    )


@serializable(recursive_serde=True)
class UserUpdate(SyftObject):
    __canonical_name__ = "UserUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID] = None

    @pydantic.validator("email", pre=True, always=True)
    def make_email(cls, v: EmailStr) -> EmailStr:
        return EmailStr(v) if v is not None else v

    email: Optional[EmailStr]
    name: Optional[str]
    role: Optional[ServiceRole] = None  # make sure role cant be set without uid
    password: Optional[str] = None
    password_verify: Optional[str] = None
    verify_key: Optional[SyftVerifyKey] = None
    institution: Optional[str] = None
    website: Optional[str] = None


@serializable(recursive_serde=True)
class UserCreate(UserUpdate):
    __canonical_name__ = "UserCreate"
    __version__ = SYFT_OBJECT_VERSION_1

    email: EmailStr
    name: str
    role: Optional[ServiceRole] = None  # make sure role cant be set without uid
    password: str
    password_verify: str
    verify_key: Optional[SyftVerifyKey] = None
    institution: Optional[str] = None
    website: Optional[str] = None

    __attr_repr_cols__ = ["name", "email"]


@serializable(recursive_serde=True)
class UserSearch(SyftObject):
    __canonical_name__ = "UserSearch"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    email: Optional[EmailStr]
    verify_key: Optional[SyftVerifyKey]
    name: Optional[str]


@serializable(recursive_serde=True)
class UserView(UserUpdate):
    __canonical_name__ = "UserView"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_repr_cols__ = ["name", "email"]


@transform(UserUpdate, User)
def user_update_to_user() -> List[Callable]:
    return [
        validate_email,
        hash_password,
        drop(["password", "password_verify"]),
    ]


@transform(UserCreate, User)
def user_create_to_user() -> List[Callable]:
    return [
        generate_id,
        validate_email,
        hash_password,
        generate_key,
        default_role(ServiceRole.GUEST),
        drop(["password", "password_verify"]),
    ]


@transform(User, UserView)
def user_to_view_user() -> List[Callable]:
    return [keep(["id", "email", "name", "role", "institution", "website"])]


@serializable(recursive_serde=True)
class UserPrivateKey(SyftObject):
    __canonical_name__ = "UserPrivateKey"
    __version__ = SYFT_OBJECT_VERSION_1

    email: str
    signing_key: SyftSigningKey


@transform(User, UserPrivateKey)
def user_to_user_verify() -> List[Callable]:
    return [keep(["email", "signing_key", "id"])]
