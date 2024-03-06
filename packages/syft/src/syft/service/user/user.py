# stdlib
from getpass import getpass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# third party
from bcrypt import checkpw
from bcrypt import gensalt
from bcrypt import hashpw
from pydantic import EmailStr
from pydantic import ValidationError
from pydantic import field_validator

# relative
from ...client.api import APIRegistry
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.syft_metaclass import Empty
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import drop
from ...types.transforms import generate_id
from ...types.transforms import keep
from ...types.transforms import make_set_default
from ...types.transforms import transform
from ...types.transforms import validate_email
from ...types.uid import UID
from ..response import SyftError
from ..response import SyftSuccess
from .user_roles import ServiceRole


@serializable()
class User(SyftObject):
    # version
    __canonical_name__ = "User"
    __version__ = SYFT_OBJECT_VERSION_2

    id: Optional[UID] = None  # type: ignore[assignment]

    # fields
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    hashed_password: Optional[str] = None
    salt: Optional[str] = None
    signing_key: Optional[SyftSigningKey] = None
    verify_key: Optional[SyftVerifyKey] = None
    role: Optional[ServiceRole] = None
    institution: Optional[str] = None
    website: Optional[str] = None
    created_at: Optional[str] = None
    # TODO where do we put this flag?
    mock_execution_permission: bool = False

    # serde / storage rules
    __attr_searchable__ = ["name", "email", "verify_key", "role"]
    __attr_unique__ = ["email", "signing_key", "verify_key"]
    __repr_attrs__ = ["name", "email"]


def default_role(role: ServiceRole) -> Callable:
    return make_set_default(key="role", value=role)


def hash_password(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context

    if context.output["password"] is not None and (
        (context.output["password_verify"] is None)
        or context.output["password"] == context.output["password_verify"]
    ):
        salt, hashed = salt_and_hash_password(context.output["password"], 12)
        context.output["hashed_password"] = hashed
        context.output["salt"] = salt

    return context


def generate_key(context: TransformContext) -> TransformContext:
    if context.output is not None:
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


@serializable()
class UserUpdate(PartialSyftObject):
    __canonical_name__ = "UserUpdate"
    __version__ = SYFT_OBJECT_VERSION_2

    @field_validator("role", mode="before")
    @classmethod
    def str_to_role(cls, v: Any) -> Any:
        if isinstance(v, str) and hasattr(ServiceRole, v.upper()):
            return getattr(ServiceRole, v.upper())
        return v

    email: EmailStr
    name: str
    role: ServiceRole  # make sure role cant be set without uid
    password: str
    password_verify: str
    verify_key: SyftVerifyKey
    institution: str
    website: str
    mock_execution_permission: bool


@serializable()
class UserCreate(UserUpdate):
    __canonical_name__ = "UserCreate"
    __version__ = SYFT_OBJECT_VERSION_2

    email: EmailStr
    name: str
    role: Optional[ServiceRole] = None  # type: ignore[assignment]
    password: str
    password_verify: Optional[str] = None  # type: ignore[assignment]
    verify_key: Optional[SyftVerifyKey] = None  # type: ignore[assignment]
    institution: Optional[str] = None  # type: ignore[assignment]
    website: Optional[str] = None  # type: ignore[assignment]
    created_by: Optional[SyftSigningKey] = None
    mock_execution_permission: bool = False

    __repr_attrs__ = ["name", "email"]


@serializable()
class UserSearch(PartialSyftObject):
    __canonical_name__ = "UserSearch"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    email: EmailStr
    verify_key: SyftVerifyKey
    name: str


@serializable()
class UserView(SyftObject):
    __canonical_name__ = "UserView"
    __version__ = SYFT_OBJECT_VERSION_2

    email: EmailStr
    name: str
    role: ServiceRole  # make sure role cant be set without uid
    institution: Optional[str] = None
    website: Optional[str] = None
    mock_execution_permission: bool

    __repr_attrs__ = ["name", "email", "institution", "website", "role"]

    def _coll_repr_(self) -> Dict[str, Any]:
        return {
            "Name": self.name,
            "Email": self.email,
            "Institute": self.institution,
            "Website": self.website,
            "Role": self.role.name.capitalize(),
        }

    def _set_password(self, new_password: str) -> Union[SyftError, SyftSuccess]:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")

        api.services.user.update(
            uid=self.id, user_update=UserUpdate(password=new_password)
        )
        return SyftSuccess(
            message=f"Successfully updated password for "
            f"user '{self.name}' with email '{self.email}'."
        )

    def set_password(
        self, new_password: Optional[str] = None, confirm: bool = True
    ) -> Union[SyftError, SyftSuccess]:
        """Set a new password interactively with confirmed password from user input"""
        # TODO: Add password validation for special characters
        if not new_password:
            new_password = getpass("New Password: ")

        if confirm:
            confirmed_password: str = getpass("Please confirm your password: ")
            if confirmed_password != new_password:
                return SyftError(message="Passwords do not match !")
        return self._set_password(new_password)

    def set_email(self, email: str) -> Union[SyftSuccess, SyftError]:
        # validate email address
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")

        try:
            user_update = UserUpdate(email=email)
        except ValidationError:
            return SyftError(message="{email} is not a valid email address.")

        result = api.services.user.update(uid=self.id, user_update=user_update)

        if isinstance(result, SyftError):
            return result

        self.email = email
        return SyftSuccess(
            message=f"Successfully updated email for the user "
            f"'{self.name}' to '{self.email}'."
        )

    def update(
        self,
        name: Union[Type[Empty], str] = Empty,
        institution: Union[Type[Empty], str] = Empty,
        website: Union[Type[Empty], str] = Empty,
        role: Union[Type[Empty], str] = Empty,
        mock_execution_permission: Union[Type[Empty], bool] = Empty,
    ) -> Union[SyftSuccess, SyftError]:
        """Used to update name, institution, website of a user."""
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(message=f"You must login to {self.node_uid}")
        user_update = UserUpdate(
            name=name,
            institution=institution,
            website=website,
            role=role,
            mock_execution_permission=mock_execution_permission,
        )
        result = api.services.user.update(uid=self.id, user_update=user_update)

        if isinstance(result, SyftError):
            return result

        for attr, val in result.to_dict(exclude_empty=True).items():
            setattr(self, attr, val)

        return SyftSuccess(message="User details successfully updated.")

    def allow_mock_execution(self, allow: bool = True) -> Union[SyftSuccess, SyftError]:
        return self.update(mock_execution_permission=allow)


@serializable()
class UserViewPage(SyftObject):
    __canonical_name__ = "UserViewPage"
    __version__ = SYFT_OBJECT_VERSION_1

    users: List[UserView]
    total: int


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
        drop(["password", "password_verify", "created_by"]),
        # TODO: Fix this by passing it from client & verifying it at server
        default_role(ServiceRole.DATA_SCIENTIST),
    ]


@transform(User, UserView)
def user_to_view_user() -> List[Callable]:
    return [
        keep(
            [
                "id",
                "email",
                "name",
                "role",
                "institution",
                "website",
                "mock_execution_permission",
            ]
        )
    ]


@serializable()
class UserPrivateKey(SyftObject):
    __canonical_name__ = "UserPrivateKey"
    __version__ = SYFT_OBJECT_VERSION_1

    email: str
    signing_key: SyftSigningKey
    role: ServiceRole


@transform(User, UserPrivateKey)
def user_to_user_verify() -> List[Callable]:
    return [keep(["email", "signing_key", "id", "role"])]
