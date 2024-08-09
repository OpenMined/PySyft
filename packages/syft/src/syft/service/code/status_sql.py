# stdlib
import uuid

# third party
import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

# syft absolute
import syft as sy

# relative
from ...types.uid import UID
from ..job.base_sql import Base
from ..job.base_sql import CommonMixin
from ..job.base_sql import PermissionMixin
from ..job.base_sql import UIDTypeDecorator
from .user_code import UserCodeStatusCollection


class UserCodeStatusCollectionDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "user_code_status_collections"

    id: Mapped[UID] = mapped_column(
        UIDTypeDecorator, primary_key=True, default=uuid.uuid4
    )
    status_dict: Mapped[bytes] = mapped_column(sa.LargeBinary)
    user_code_id: Mapped[UID | None] = mapped_column(
        UIDTypeDecorator, ForeignKey("user_codes.id"), nullable=True
    )
    user_code = relationship("UserCodeDB", back_populates="status_collection")
    user_code_link: Mapped[bytes]

    @classmethod
    def from_obj(cls, obj: UserCodeStatusCollection) -> "UserCodeStatusCollectionDB":
        return cls(
            id=obj.id,
            status_dict=sy.serialize(obj.status_dict, to_bytes=True),
            user_code_link=sy.serialize(obj.user_code_link, to_bytes=True),
        )

    def to_obj(self) -> UserCodeStatusCollection:
        return UserCodeStatusCollection(
            id=self.id,
            status_dict=sy.deserialize(self.status_dict, from_bytes=True),
            user_code_link=sy.deserialize(self.user_code_link, from_bytes=True),
        )


UserCodeStatusCollectionDB._init_perms()
