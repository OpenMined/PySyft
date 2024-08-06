# third party
import uuid
import sqlalchemy as sa
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from syft.abstract_server import ServerSideType
from syft.server.credentials import SyftVerifyKey
from syft.service.action.action_permissions import ActionPermission
from syft.service.code.user_code import UserCode, UserCodeStatusCollection
from sqlalchemy.orm import relationship

# third party
from result import Err
import sqlalchemy as sa
from sqlalchemy import Column, Enum, ForeignKey, String, Table
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON
from syft.service.action.action_permissions import ActionPermission
from syft.types.syft_object import SyftObject
from typing_extensions import Self, TypeVar

# syft absolute
import syft as sy
from syft.service.job.job_sql import (
    Base,
    CommonMixin,
    PermissionMixin,
    unwrap_uid,
    wrap_uid,
)
from syft.types.datetime import DateTime
from syft.types.uid import UID


class UserCodeStatusCollectionDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "user_code_status_collections"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    status_dict: Mapped[bytes] = mapped_column(sa.LargeBinary)
    user_code_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.Uuid, ForeignKey("user_codes.id"), nullable=True
    )
    user_code = relationship("UserCodeDB", back_populates="status_collection")
    user_code_link: Mapped[bytes]

    @classmethod
    def from_obj(cls, obj: UserCodeStatusCollection) -> "UserCodeStatusCollectionDB":
        return cls(
            id=unwrap_uid(obj.id),
            status_dict=sy.serialize(obj.status_dict, to_bytes=True),
            user_code_link=sy.serialize(obj.user_code_link, to_bytes=True),
        )

    def to_obj(self) -> UserCodeStatusCollection:
        return UserCodeStatusCollection(
            id=wrap_uid(self.id),
            status_dict=sy.deserialize(self.status_dict, from_bytes=True),
            user_code_link=sy.deserialize(self.user_code_link, from_bytes=True),
        )


UserCodeStatusCollectionDB._init_perms()
