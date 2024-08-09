# stdlib
from dataclasses import dataclass, field
from re import T
from typing import TYPE_CHECKING, Generic
import uuid

# third party
from colorama import Fore
import pydantic
from result import Err
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import TypeDecorator
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import registry
from sqlalchemy.orm import relationship
from sqlalchemy.types import Enum
from sqlalchemy.types import JSON
from syft.store.linked_obj import LinkedObject
from typing_extensions import Self
from typing_extensions import TypeVar

# syft absolute
import syft as sy

# relative
from ...server.credentials import SyftVerifyKey
from ...types.datetime import DateTime
from ...types.syft_object import SyftObject
from ...types.syft_object_registry import SyftObjectRegistry
from ...types.uid import LineageID
from ...types.uid import UID
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..action.action_object import ActionType
from ..action.action_object import TwinMode
from ..action.action_permissions import ActionPermission
from .job_stash import Job
from .job_stash import JobStatus
from .job_stash import JobType

if TYPE_CHECKING:
    from syft.service.log.log_sql import SyftLogDB
    from syft.service.code.user_code_sql import UserCodeDB
    from syft.service.output.execution_output_sql import ExecutionOutputDB


mapper_registry = registry()


class UIDTypeDecorator(TypeDecorator):
    """Converts between Syft UID and UUID."""

    impl = sa.UUID
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return UID(value)


class VerifyKeyTypeDecorator(TypeDecorator):
    """Converts between Syft VerifyKey and str."""

    impl = sa.String
    cache_ok = True

    def process_bind_param(self, value: SyftVerifyKey, dialect):
        if value is not None:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is not None:
            return SyftVerifyKey(value)


class DateTimeTypeDecorator(TypeDecorator):
    """Converts between Syft DateTime and datetime."""

    impl = sa.DateTime
    cache_ok = True

    def process_bind_param(self, value: DateTime, dialect):
        if value is not None:
            return value.to_datetime()

    def process_result_value(self, value, dialect):
        if value is not None:
            return DateTime.from_datetime(value)


class Base(DeclarativeBase):
    ignore_fields = ["created_at", "modified_by", "updated_at"]

    def to_dict(self):
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if column.name not in self.ignore_fields
        }

    def update_obj(self, other: type[Self]) -> None:
        for key, value in other.to_dict().items():
            if key not in self.ignore_fields:
                setattr(self, key, value)


class CommonMixin:
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    id: Mapped[UID] = mapped_column(
        UIDTypeDecorator,
        default=uuid.uuid4,
        primary_key=True,
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTimeTypeDecorator, server_default=sa.func.now()
    )

    updated_at: Mapped[DateTime] = mapped_column(
        DateTimeTypeDecorator,
        server_default=sa.func.now(),
        server_onupdate=sa.func.now(),
    )
    deleted_at: Mapped[DateTime | None] = mapped_column(
        DateTimeTypeDecorator, default=None
    )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.short_id()})"

    def short_id(self) -> str:
        return str(self.id.hex)[:8]

    def __repr__(self) -> str:
        # show all columns indented on new lines
        return "\n".join(
            [
                f"{self.__class__.__name__}(",
                *[
                    f"    {col}: {getattr(self, col)}"
                    for col in self.__table__.columns.keys()
                ],
                ")",
            ]
        )


def wrap_lineage_id(uid: uuid.UUID | None) -> LineageID | None:
    if uid is None:
        return None
    return LineageID(value=uid)


ObjectT = TypeVar("ObjectT", bound=SyftObject)
PermissionT = TypeVar("PermissionT", bound=Base)


class BaseSchema(Generic[ObjectT, PermissionT]):
    id: UID

    def to_obj(self) -> ObjectT:
        raise NotImplementedError

    @classmethod
    def from_obj(cls, obj: ObjectT) -> Self:
        raise NotImplementedError


SchemaT = TypeVar("SchemaT", bound=BaseSchema)


@dataclass
class PermissionABC:
    """Used for type hinting and binding to SQLAlchemy."""

    object_id: UID
    user_id: str | None
    permission: ActionPermission
    id: uuid.UUID = field(default_factory=UID)


class PermissionMixin:
    @declared_attr
    def permissions_table(cls):
        return Table(
            f"{cls.__tablename__}_permissions",
            Base.metadata,
            Column("id", sa.Uuid, primary_key=True, default=uuid.uuid4),
            Column("object_id", sa.Uuid, ForeignKey(f"{cls.__tablename__}.id")),
            Column("user_id", String, default=None),
            Column("permission", Enum(ActionPermission)),
        )

    @declared_attr
    def PermissionModel(cls):
        return type(
            f"{str(cls.__tablename__).title()}PermissionModel",
            (PermissionABC,),
            {},
        )

    @classmethod
    def _init_perms(cls):
        """Map the model to the table and create the relationship to the permissions."""
        mapper_registry.map_imperatively(cls.PermissionModel, cls.permissions_table)
        cls.permissions: Mapped[list[cls.PermissionModel]] = relationship(
            cls.PermissionModel, lazy="joined"
        )
