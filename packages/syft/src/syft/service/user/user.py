# stdlib
from collections.abc import Callable
from datetime import datetime
from getpass import getpass
import re
from typing import Any

# third party
from bcrypt import checkpw
from bcrypt import gensalt
from bcrypt import hashpw
from pydantic import EmailStr
from pydantic import ValidationError

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...types.errors import SyftException
from ...types.syft_metaclass import Empty
from ...types.syft_migration import migrate
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
from ..notifier.notifier_enums import NOTIFIERS
from ..response import SyftSuccess
from .errors import UserPasswordMismatchError
from .user_roles import ServiceRole


@serializable()
class UserV1(SyftObject):
    # version
    __canonical_name__ = "User"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore[assignment]

    # fields
    notifications_enabled: dict[NOTIFIERS, bool] = {
        NOTIFIERS.EMAIL: True,
        NOTIFIERS.SMS: False,
        NOTIFIERS.SLACK: False,
        NOTIFIERS.APP: False,
    }
    email: EmailStr | None = None
    name: str | None = None
    hashed_password: str | None = None
    salt: str | None = None
    signing_key: SyftSigningKey | None = None
    verify_key: SyftVerifyKey | None = None
    role: ServiceRole | None = None
    institution: str | None = None
    website: str | None = None
    created_at: str | None = None
    # TODO where do we put this flag?
    mock_execution_permission: bool = False

    # serde / storage rules
    __attr_searchable__ = ["name", "email", "verify_key", "role"]
    __attr_unique__ = ["email", "signing_key", "verify_key"]
    __repr_attrs__ = ["name", "email"]


@serializable()
class User(SyftObject):
    # version
    __canonical_name__ = "User"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID | None = None  # type: ignore[assignment]

    # fields
    notifications_enabled: dict[NOTIFIERS, bool] = {
        NOTIFIERS.EMAIL: True,
        NOTIFIERS.SMS: False,
        NOTIFIERS.SLACK: False,
        NOTIFIERS.APP: False,
    }
    email: EmailStr | None = None
    name: str | None = None
    hashed_password: str | None = None
    salt: str | None = None
    signing_key: SyftSigningKey | None = None
    verify_key: SyftVerifyKey | None = None
    role: ServiceRole | None = None
    institution: str | None = None
    website: str | None = None
    created_at: str | None = None
    # TODO where do we put this flag?
    mock_execution_permission: bool = False
    reset_token: str | None = None
    reset_token_date: datetime | None = None
    # serde / storage rules
    __attr_searchable__ = ["name", "email", "verify_key", "role", "reset_token"]
    __attr_unique__ = ["email", "signing_key", "verify_key"]
    __repr_attrs__ = ["name", "email"]


@migrate(UserV1, User)
def migrate_server_user_update_v1_current() -> list[Callable]:
    return [
        make_set_default("reset_token", None),
        make_set_default("reset_token_date", None),
        drop("__attr_searchable__"),
        make_set_default(
            "__attr_searchable__",
            ["name", "email", "verify_key", "role", "reset_token"],
        ),
    ]


@migrate(User, UserV1)
def migrate_server_user_downgrade_current_v1() -> list[Callable]:
    return [
        drop("reset_token"),
        drop("reset_token_date"),
        drop("__attr_searchable__"),
        make_set_default(
            "__attr_searchable__", ["name", "email", "verify_key", "role"]
        ),
    ]


def default_role(role: ServiceRole) -> Callable:
    return make_set_default(key="role", value=role)


def validate_password(password: str) -> bool:
    # Define the regex pattern for the password
    pattern = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$")

    return bool(pattern.match(password))


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


def salt_and_hash_password(password: str, rounds: int) -> tuple[str, str]:
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
    __version__ = SYFT_OBJECT_VERSION_1

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
class UserCreate(SyftObject):
    __canonical_name__ = "UserCreate"
    __version__ = SYFT_OBJECT_VERSION_1

    email: EmailStr
    name: str
    role: ServiceRole | None = None  # type: ignore[assignment]
    password: str
    password_verify: str | None = None  # type: ignore[assignment]
    verify_key: SyftVerifyKey | None = None  # type: ignore[assignment]
    institution: str | None = ""  # type: ignore[assignment]
    website: str | None = ""  # type: ignore[assignment]
    created_by: SyftSigningKey | None = None  # type: ignore[assignment]
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
    __version__ = SYFT_OBJECT_VERSION_1

    notifications_enabled: dict[NOTIFIERS, bool] = {
        NOTIFIERS.EMAIL: True,
        NOTIFIERS.SMS: False,
        NOTIFIERS.SLACK: False,
        NOTIFIERS.APP: False,
    }
    email: EmailStr
    name: str
    role: ServiceRole  # make sure role cant be set without uid
    institution: str | None = None
    website: str | None = None
    mock_execution_permission: bool

    __repr_attrs__ = [
        "name",
        "email",
        "institution",
        "website",
        "role",
        "notifications_enabled",
    ]

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "Name": self.name,
            "Email": self.email,
            "Institute": self.institution,
            "Website": self.website,
            "Role": self.role.name.capitalize(),
            "Notifications": "Email: "
            + (
                "Enabled" if self.notifications_enabled[NOTIFIERS.EMAIL] else "Disabled"
            ),
        }

    def _set_password(self, new_password: str) -> SyftSuccess:
        client = self.get_api()

        client.services.user.update(uid=self.id, password=new_password)

        return SyftSuccess(
            message=f"Successfully updated password for user '{self.email}'."
        )

    def set_password(
        self, new_password: str | None = None, confirm: bool = True
    ) -> SyftSuccess:
        """Set a new password interactively with confirmed password from user input"""
        # TODO: Add password validation for special characters
        if not new_password:
            new_password = getpass("New Password: ")

        if confirm:
            confirmed_password: str = getpass("Please confirm your password: ")
            if confirmed_password != new_password:
                raise UserPasswordMismatchError

        return self._set_password(new_password)

    def set_email(self, email: str) -> SyftSuccess:
        try:
            user_update = UserUpdate(email=email)
        except ValidationError:
            raise SyftException(public_message=f"Invalid email: '{email}'.")

        api = self.get_api()

        # TODO: Shouldn't this trigger an update on self?
        result = api.services.user.update(uid=self.id, email=user_update.email)

        return SyftSuccess(message=f"Email updated to '{result.email}'.")

    def update(
        self,
        name: type[Empty] | str = Empty,
        institution: type[Empty] | str = Empty,
        website: type[Empty] | str = Empty,
        role: type[Empty] | str = Empty,
        mock_execution_permission: type[Empty] | bool = Empty,
    ) -> SyftSuccess:
        """Used to update name, institution, website of a user."""
        api = self.get_api()

        result = api.services.user.update(
            uid=self.id,
            name=name,
            institution=institution,
            website=website,
            role=role,
            mock_execution_permission=mock_execution_permission,
        )

        for attr, val in result.to_dict(exclude_empty=True).items():
            setattr(self, attr, val)

        return SyftSuccess(message="User details successfully updated.")

    def allow_mock_execution(self, allow: bool = True) -> SyftSuccess:
        return self.update(mock_execution_permission=allow)


@serializable()
class UserViewPage(SyftObject):
    __canonical_name__ = "UserViewPage"
    __version__ = SYFT_OBJECT_VERSION_1

    users: list[UserView]
    total: int


@transform(UserUpdate, User)
def user_update_to_user() -> list[Callable]:
    return [
        validate_email,
        hash_password,
        drop(["password", "password_verify"]),
    ]


@transform(UserCreate, User)
def user_create_to_user() -> list[Callable]:
    return [
        generate_id,
        validate_email,
        hash_password,
        generate_key,
        drop(["password", "password_verify", "created_by"]),
        # TODO: Fix this by passing it from client & verifying it at server
        default_role(ServiceRole.DATA_SCIENTIST),
    ]


@transform(UserV1, UserView)
def userv1_to_view_user() -> list[Callable]:
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
                "notifications_enabled",
            ]
        )
    ]


@transform(User, UserView)
def user_to_view_user() -> list[Callable]:
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
                "notifications_enabled",
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


@transform(UserV1, UserPrivateKey)
def userv1_to_user_verify() -> list[Callable]:
    return [keep(["email", "signing_key", "id", "role"])]


@transform(User, UserPrivateKey)
def user_to_user_verify() -> list[Callable]:
    return [keep(["email", "signing_key", "id", "role"])]
