# third party
import uuid
import sqlalchemy as sa

import syft as sy

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from syft.server.credentials import SyftSigningKey, SyftVerifyKey
from syft.service.job.job_sql import (
    Base,
    CommonMixin,
    PermissionMixin,
    unwrap_uid,
    wrap_uid,
)
from syft.service.notifier.notifier_enums import NOTIFIERS
from syft.service.user.user import User
from syft.service.user.user_roles import ServiceRole

default_notifications = {
    NOTIFIERS.EMAIL: True,
    NOTIFIERS.SMS: False,
    NOTIFIERS.SLACK: False,
    NOTIFIERS.APP: False,
}


class UserDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    notifications_enabled: Mapped[bytes]
    email: Mapped[str | None]
    name: Mapped[str | None]
    hashed_password: Mapped[str | None]
    salt: Mapped[str | None]
    signing_key: Mapped[str | None]
    verify_key: Mapped[str | None]
    role: Mapped[ServiceRole | None]
    institution: Mapped[str | None]
    website: Mapped[str | None]
    mock_execution_permission: Mapped[bool]

    @classmethod
    @classmethod
    def from_obj(cls, obj: "User") -> "UserDB":
        return cls(
            id=unwrap_uid(obj.id),
            notifications_enabled=sy.serialize(default_notifications, to_bytes=True),
            email=obj.email,
            name=obj.name,
            hashed_password=obj.hashed_password,
            salt=obj.salt,
            signing_key=str(obj.signing_key),
            verify_key=str(obj.verify_key),
            role=obj.role,
            institution=obj.institution,
            website=obj.website,
            mock_execution_permission=obj.mock_execution_permission,
        )

    def to_obj(self) -> "User":
        return User(
            id=wrap_uid(self.id),
            notifications_enabled=sy.deserialize(
                self.notifications_enabled, from_bytes=True
            ),
            email=self.email,
            name=self.name,
            hashed_password=self.hashed_password,
            salt=self.salt,
            signing_key=SyftSigningKey.from_string(self.signing_key),
            verify_key=SyftVerifyKey.from_string(self.verify_key),
            role=self.role,
            institution=self.institution,
            website=self.website,
            mock_execution_permission=self.mock_execution_permission,
        )


UserDB._init_perms()
