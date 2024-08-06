# stdlib
from datetime import datetime
from typing import Any, Generic
import uuid
from sqlalchemy.types import Enum

# third party
from result import Err
import sqlalchemy as sa
from sqlalchemy import Column, ForeignKey, String, Table
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship, mapper
from sqlalchemy.types import JSON
from syft.service.action.action_permissions import ActionPermission
from syft.types.syft_object import SyftObject
from typing_extensions import Self, TypeVar

# syft absolute
import syft as sy

# relative
from ...types.datetime import DateTime
from ...types.syft_object_registry import SyftObjectRegistry
from ...types.uid import LineageID
from ...types.uid import UID
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..action.action_object import ActionType
from ..action.action_object import TwinMode
from .job_stash import Job
from .job_stash import JobStatus
from .job_stash import JobType
from sqlalchemy.orm import registry

mapper_registry = registry()


class Base(DeclarativeBase):
    def to_dict(self):
        ignore_fields = ["created_at", "modified_at", "modified_by"]
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if column.name not in ignore_fields
        }

    def update(self, other: type[Self]) -> None:
        for key, value in other.to_dict().items():
            setattr(self, key, value)


class CommonMixin:
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    # id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(server_default=sa.func.now())
    modified_at: Mapped[datetime] = mapped_column(
        server_default=sa.func.now(), server_onupdate=sa.func.now()
    )
    modified_by: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid)

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


def unwrap_uid(uid: UID | None) -> str | None:
    if uid is None:
        return None
    return uid.value


def wrap_uid(uid: uuid.UUID | None) -> UID | None:
    if uid is None:
        return None
    return UID(value=uid)


def wrap_lineage_id(uid: uuid.UUID | None) -> LineageID | None:
    if uid is None:
        return None
    return LineageID(value=uid)


ObjectT = TypeVar("ObjectT", bound=SyftObject)
PermissionT = TypeVar("PermissionT", bound=Base)


class BaseSchema(Generic[ObjectT, PermissionT]):
    def to_obj(self) -> ObjectT:
        raise NotImplementedError

    @classmethod
    def from_obj(cls, obj: ObjectT) -> Self:
        raise NotImplementedError


SchemaT = TypeVar("SchemaT", bound=BaseSchema)


_tablename_ = "jobs"
_job_permissions_tablename_ = "job_permissions"


from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy.ext.declarative import declared_attr


def init(self, *args, **kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)


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
            f"{cls.__tablename__}PermissionModel",
            (object,),
            {"object_id": None, "user_id": None, "permission": None, "__init__": init},
        )

    @classmethod
    def _init_perms(cls):
        mapper_registry.map_imperatively(cls.PermissionModel, cls.permissions_table)
        cls.permissions: Mapped[list[cls.PermissionModel]] = relationship(
            cls.PermissionModel, lazy="joined"
        )


class JobDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)

    server_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    result_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("actionobjectdb.id"), default=None
    )
    result: Mapped["ActionObjectDB | None"] = relationship(
        "ActionObjectDB", lazy="joined"
    )
    result_error: Mapped[str | None] = mapped_column(default=None)
    resolved: Mapped[bool] = mapped_column(default=False)
    status: Mapped[JobStatus] = mapped_column(default=JobStatus.CREATED)
    n_iters: Mapped[int | None] = mapped_column(default=None)
    current_iter: Mapped[int | None] = mapped_column(default=None)
    action_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("actiondb.id"), default=None
    )
    action: Mapped["ActionDB | None"] = relationship("ActionDB", lazy="joined")

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{__tablename__}.id"), default=None
    )
    children: Mapped[list["JobDB"]] = relationship(
        "JobDB",
        back_populates="parent",
        remote_side=[id],
    )
    parent: Mapped["JobDB"] = relationship(
        back_populates="children",
    )

    job_pid: Mapped[int | None] = mapped_column(default=None)
    job_worker_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    log_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    user_code_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    requested_by: Mapped[uuid.UUID | None] = mapped_column(default=None)
    job_type: Mapped[JobType] = mapped_column(default=JobType.JOB)

    @classmethod
    def from_obj(cls, obj: "Job") -> "JobDB":
        return cls(
            id=unwrap_uid(obj.id),
            server_uid=unwrap_uid(obj.server_uid),
            result_id=unwrap_uid(obj.result_id),
            result=ActionObjectDB.from_obj(obj.result)
            if isinstance(obj.result, ActionObject)
            else None,
            result_error=obj.result.value if isinstance(obj.result, Err) else None,
            resolved=obj.resolved,
            status=obj.status,
            n_iters=obj.n_iters,
            current_iter=obj.current_iter,
            action=ActionDB.from_obj(obj.action) if obj.action is not None else None,
            parent_id=unwrap_uid(obj.parent_job_id),
            job_pid=obj.job_pid,
            job_worker_id=unwrap_uid(obj.job_worker_id),
            log_id=unwrap_uid(obj.log_id),
            user_code_id=unwrap_uid(obj.user_code_id),
            requested_by=unwrap_uid(obj.requested_by),
            job_type=obj.job_type,
        )

    def to_obj(self) -> Job:
        if self.result is not None:
            result = self.result.to_obj()
        elif self.result_error is not None:
            result = Err(self.result_error)
        else:
            result = None

        return Job(
            id=wrap_uid(self.id),
            server_uid=wrap_uid(self.server_uid),
            result_id=wrap_uid(self.result_id),
            result=result,
            resolved=self.resolved,
            status=self.status,
            n_iters=self.n_iters,
            current_iter=self.current_iter,
            action=self.action.to_obj() if self.action is not None else None,
            parent_job_id=wrap_uid(self.parent_id),
            job_pid=self.job_pid,
            job_worker_id=wrap_uid(self.job_worker_id),
            log_id=wrap_uid(self.log_id),
            user_code_id=wrap_uid(self.user_code_id),
            requested_by=wrap_uid(self.requested_by),
            job_type=self.job_type,
        )


JobDB._init_perms()


class ActionDB(CommonMixin, Base):
    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    path: Mapped[str | None] = mapped_column(default=None)
    op: Mapped[str | None] = mapped_column(default=None)
    remote_self: Mapped[uuid.UUID | None] = mapped_column(default=None)
    args: Mapped[list[str]] = mapped_column(JSON, default=[])
    kwargs: Mapped[dict[str, str]] = mapped_column(JSON, default={})
    result_id: Mapped[uuid.UUID] = mapped_column()
    action_type: Mapped[ActionType | None] = mapped_column(default=None)
    user_code_id: Mapped[uuid.UUID | None] = mapped_column(default=None)

    @classmethod
    def from_obj(cls, obj: Action) -> Self:
        return cls(
            id=unwrap_uid(obj.id),
            path=obj.path,
            op=obj.op,
            remote_self=unwrap_uid(obj.remote_self),
            args=[arg.value.hex for arg in obj.args],
            kwargs={k: v.value.hex for k, v in obj.kwargs.items()},
            result_id=unwrap_uid(obj.result_id),
            action_type=obj.action_type,
            user_code_id=unwrap_uid(obj.user_code_id),
        )

    def to_obj(self) -> Action:
        return Action(
            id=wrap_uid(self.id),
            path=self.path,
            op=self.op,
            remote_self=wrap_lineage_id(self.remote_self),
            args=[wrap_lineage_id(arg) for arg in self.args],
            kwargs={k: wrap_lineage_id(v) for k, v in self.kwargs.items()},
            result_id=wrap_lineage_id(self.result_id),
            action_type=self.action_type,
            user_code_id=wrap_uid(self.user_code_id),
        )


# class ActionObject(SyncableSyftObject):
#     """Action object for remote execution."""

#     __canonical_name__ = "ActionObject"
#     __version__ = SYFT_OBJECT_VERSION_1
#     __private_sync_attr_mocks__: ClassVar[dict[str, Any]] = {
#         "syft_action_data_cache": None,
#         "syft_blob_storage_entry_id": None,
#     }

#     __attr_searchable__: list[str] = []  # type: ignore[misc]
#     syft_action_data_cache: Any | None = None
#     syft_blob_storage_entry_id: UID | None = None
#     syft_pointer_type: ClassVar[type[ActionObjectPointer]]

#     # Help with calculating history hash for code verification
#     syft_parent_hashes: int | list[int] | None = None
#     syft_parent_op: str | None = None
#     syft_parent_args: Any | None = None
#     syft_parent_kwargs: Any | None = None
#     syft_history_hash: int | None = None
#     syft_internal_type: ClassVar[type[Any]]
#     syft_server_uid: UID | None = None
#     syft_pre_hooks__: dict[str, list] = {}
#     syft_post_hooks__: dict[str, list] = {}
#     syft_twin_type: TwinMode = TwinMode.NONE
#     syft_passthrough_attrs: list[str] = BASE_PASSTHROUGH_ATTRS
#     syft_action_data_type: type | None = None
#     syft_action_data_repr_: str | None = None
#     syft_action_data_str_: str | None = None
#     syft_has_bool_attr: bool | None = None
#     syft_resolve_data: bool | None = None
#     syft_created_at: DateTime | None = None
#     syft_resolved: bool = True
#     syft_action_data_server_id: UID | None = None
#     syft_action_saved_to_blob_store: bool = True
#     # syft_dont_wrap_attrs = ["shape"]


class ActionObjectDB(CommonMixin, Base):
    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    syft_blob_storage_entry_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    syft_action_data_cache: Mapped[bytes | None] = mapped_column(default=None)
    syft_history_hash: Mapped[int | None] = mapped_column(default=None)
    syft_server_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    syft_twin_type: Mapped[TwinMode] = mapped_column()
    syft_action_data_type: Mapped[str | None] = mapped_column(default=None)
    syft_action_data_repr_: Mapped[str | None] = mapped_column(default=None)
    syft_action_data_str_: Mapped[str | None] = mapped_column(default=None)
    syft_has_bool_attr: Mapped[bool] = mapped_column(default=None)
    syft_resolved: Mapped[bool] = mapped_column(default=True)
    syft_action_data_server_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    syft_action_saved_to_blob_store: Mapped[bool] = mapped_column(default=True)

    @classmethod
    def from_obj(cls, obj: ActionObject) -> "ActionObjectDB":
        if obj.syft_action_data_type:
            action_data_type = SyftObjectRegistry.__type_to_canonical_name__[
                obj.syft_action_data_type
            ][0]
        else:
            action_data_type = None
        return cls(
            id=unwrap_uid(obj.id),
            syft_blob_storage_entry_id=unwrap_uid(obj.syft_blob_storage_entry_id),
            # syft_action_data_cache=sy.serialize(
            #     obj.syft_action_data_cache, to_bytes=True
            # )
            # if obj.syft_action_data_cache is not None
            # else None,
            syft_history_hash=obj.syft_history_hash,
            syft_server_uid=unwrap_uid(obj.syft_server_uid),
            syft_twin_type=obj.syft_twin_type,
            syft_action_data_type=action_data_type,
            syft_action_data_repr_=obj.syft_action_data_repr_,
            syft_action_data_str_=obj.syft_action_data_str_,
            syft_has_bool_attr=obj.syft_has_bool_attr,
            syft_resolved=obj.syft_resolved,
            created_at=obj.syft_created_at.to_datetime()
            if obj.syft_created_at
            else None,
            syft_action_data_server_id=unwrap_uid(obj.syft_action_data_server_id)
            if obj.syft_action_data_server_id
            else None,
            syft_action_saved_to_blob_store=obj.syft_action_saved_to_blob_store,
        )

    def to_obj(self) -> ActionObject:
        if self.syft_action_data_type:
            dtype = SyftObjectRegistry.get_serde_class(self.syft_action_data_type, 1)
        else:
            dtype = None

        return ActionObject(
            id=wrap_uid(self.id),
            syft_blob_storage_entry_id=wrap_uid(self.syft_blob_storage_entry_id),
            syft_action_data_cache=sy.deserialize(
                self.syft_action_data_cache, from_bytes=True
            )
            if self.syft_action_data_cache is not None
            else None,
            syft_history_hash=self.syft_history_hash,
            syft_server_uid=wrap_uid(self.syft_server_uid),
            syft_twin_type=self.syft_twin_type,
            syft_action_data_type=dtype,
            syft_action_data_repr_=self.syft_action_data_repr_,
            syft_action_data_str_=self.syft_action_data_str_,
            syft_has_bool_attr=self.syft_has_bool_attr,
            syft_resolved=self.syft_resolved,
            created_at=DateTime.from_datetime(self.created_at)
            if self.created_at
            else None,
            syft_action_data_server_id=wrap_uid(self.syft_action_data_server_id),
            syft_action_saved_to_blob_store=self.syft_action_saved_to_blob_store,
        )
