# stdlib
from datetime import datetime
import uuid

# third party
import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON

# syft absolute
import syft as sy
from syft.service.action.action_object import Action, ActionType

# relative
from ...types.uid import UID, LineageID
from .job_stash import Job
from .job_stash import JobStatus
from .job_stash import JobType

from typing_extensions import Self


class Base(DeclarativeBase):
    def to_dict(self):
        ignore_fields = ["created_at", "modified_at", "modified_by"]
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if column.name not in ignore_fields
        }


class CommonMixin:
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, default=uuid.uuid4)
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


class JobDB(CommonMixin, Base):
    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    server_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    result_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    result: Mapped[bytes | None] = mapped_column(default=None)
    resolved: Mapped[bool] = mapped_column(default=False)
    status: Mapped[JobStatus] = mapped_column(default=JobStatus.CREATED)
    n_iters: Mapped[int | None] = mapped_column(default=None)
    current_iter: Mapped[int | None] = mapped_column(default=None)
    action_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("actiondb.id"), default=None
    )
    action: Mapped["ActionDB | None"] = relationship("ActionDB", lazy="joined")

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobdb.id"), default=None
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
            result=sy.serialize(obj.result, to_bytes=True),
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
        return Job(
            id=wrap_uid(self.id),
            server_uid=wrap_uid(self.server_uid),
            result_id=wrap_uid(self.result_id),
            result=sy.deserialize(self.result, from_bytes=True),
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


# Needed Action fields
# path: str | None = None
# op: str | None = None
# remote_self: LineageID | None = None
# args: list[LineageID]
# kwargs: dict[str, LineageID]
# result_id: LineageID = Field(default_factory=lambda: LineageID(UID()))
# action_type: ActionType | None = None
# create_object: SyftObject | None = None
# user_code_id: UID | None = None


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
