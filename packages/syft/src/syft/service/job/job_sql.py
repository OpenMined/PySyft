# stdlib
from re import T
from typing import TYPE_CHECKING, Generic
import uuid

# third party
from result import Err
from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import registry
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON
from syft.service.job.base_sql import (
    Base,
    CommonMixin,
    PermissionMixin,
    UIDTypeDecorator,
    wrap_lineage_id,
)
from typing_extensions import Self

# syft absolute
import syft as sy

# relative
from ...types.syft_object_registry import SyftObjectRegistry
from ...types.uid import UID
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..action.action_object import ActionType
from ..action.action_object import TwinMode
from .job_stash import Job
from .job_stash import JobStatus
from .job_stash import JobType

if TYPE_CHECKING:
    from syft.service.log.log_sql import SyftLogDB
    from syft.service.code.user_code_sql import UserCodeDB
    from syft.service.output.execution_output_sql import ExecutionOutputDB


mapper_registry = registry()


class JobDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "jobs"
    server_uid: Mapped[UID | None] = mapped_column(UIDTypeDecorator, default=None)
    result_id: Mapped[UID | None] = mapped_column(
        UIDTypeDecorator, ForeignKey("actionobjectdb.id"), default=None
    )
    result: Mapped["ActionObjectDB | None"] = relationship(
        "ActionObjectDB", lazy="joined"
    )
    result_error: Mapped[str | None] = mapped_column(default=None)
    resolved: Mapped[bool] = mapped_column(default=False)
    status: Mapped[JobStatus] = mapped_column(default=JobStatus.CREATED)
    n_iters: Mapped[int | None] = mapped_column(default=None)
    current_iter: Mapped[int | None] = mapped_column(default=None)
    action_id: Mapped[UID | None] = mapped_column(
        UIDTypeDecorator, ForeignKey("actiondb.id"), default=None
    )
    action: Mapped["ActionDB | None"] = relationship("ActionDB", lazy="joined")

    parent_id: Mapped[UID | None] = mapped_column(
        UIDTypeDecorator, ForeignKey(f"{__tablename__}.id"), default=None
    )

    parent: Mapped["JobDB"] = relationship(
        back_populates="children",
    )

    @declared_attr
    def children(cls) -> Mapped[list["JobDB"]]:
        return relationship(
            "JobDB",
            back_populates="parent",
            remote_side=[cls.id],
        )

    job_pid: Mapped[int | None] = mapped_column(UIDTypeDecorator, default=None)
    job_worker_id: Mapped[UID | None] = mapped_column(UIDTypeDecorator, default=None)

    log: Mapped["SyftLogDB"] = relationship(
        back_populates="job", uselist=True, lazy="joined"
    )
    user_code_id: Mapped[UID | None] = mapped_column(
        UIDTypeDecorator,
        ForeignKey("user_codes.id"),
        default=None,
    )
    user_code: Mapped["UserCodeDB"] = relationship(
        "UserCodeDB", uselist=True, lazy="joined"
    )
    requested_by: Mapped[UID | None] = mapped_column(UIDTypeDecorator, default=None)
    job_type: Mapped[JobType] = mapped_column(default=JobType.JOB)

    execution_output: Mapped["ExecutionOutputDB"] = relationship(
        "ExecutionOutputDB", uselist=True, lazy="joined"
    )

    @classmethod
    def from_obj(cls, obj: "Job") -> "JobDB":
        return cls(
            id=obj.id,
            server_uid=obj.server_uid,
            result_id=obj.result_id,
            result=ActionObjectDB.from_obj(obj.result)
            if isinstance(obj.result, ActionObject)
            else None,
            result_error=obj.result.value if isinstance(obj.result, Err) else None,
            resolved=obj.resolved,
            status=obj.status,
            n_iters=obj.n_iters,
            current_iter=obj.current_iter,
            action=ActionDB.from_obj(obj.action) if obj.action is not None else None,
            parent_id=obj.parent_job_id,
            job_pid=obj.job_pid,
            job_worker_id=obj.job_worker_id,
            # log_id=obj.log_id,
            # log=SyftLogDB.from_obj(obj.log) if obj.log is not None else None,
            user_code_id=obj.user_code_id,
            requested_by=obj.requested_by,
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
            id=self.id,
            server_uid=self.server_uid,
            result_id=self.result_id,
            result=result,
            resolved=self.resolved,
            status=self.status,
            n_iters=self.n_iters,
            current_iter=self.current_iter,
            action=self.action.to_obj() if self.action is not None else None,
            parent_job_id=self.parent_id,
            job_pid=self.job_pid,
            job_worker_id=self.job_worker_id,
            log_id=self.log.id if self.log is not None else None,
            user_code_id=self.user_code.id if self.user_code is not None else None,
            requested_by=self.requested_by,
            job_type=self.job_type,
        )


JobDB._init_perms()


class ActionDB(CommonMixin, Base):
    id: Mapped[UID] = mapped_column(
        UIDTypeDecorator, primary_key=True, default=uuid.uuid4
    )
    path: Mapped[str | None] = mapped_column(default=None)
    op: Mapped[str | None] = mapped_column(default=None)
    remote_self: Mapped[UID | None] = mapped_column(UIDTypeDecorator, default=None)
    args: Mapped[list[str]] = mapped_column(JSON, default=[])
    kwargs: Mapped[dict[str, str]] = mapped_column(JSON, default={})
    result_id: Mapped[UID] = mapped_column(UIDTypeDecorator)
    action_type: Mapped[ActionType | None] = mapped_column(default=None)
    user_code_id: Mapped[UID | None] = mapped_column(UIDTypeDecorator, default=None)

    @classmethod
    def from_obj(cls, obj: Action) -> Self:
        return cls(
            id=obj.id,
            path=obj.path,
            op=obj.op,
            remote_self=obj.remote_self,
            args=[arg.value.hex for arg in obj.args],
            kwargs={k: v.value.hex for k, v in obj.kwargs.items()},
            result_id=obj.result_id,
            action_type=obj.action_type,
            user_code_id=obj.user_code_id,
        )

    def to_obj(self) -> Action:
        return Action(
            id=self.id,
            path=self.path,
            op=self.op,
            remote_self=wrap_lineage_id(self.remote_self),
            args=[wrap_lineage_id(arg) for arg in self.args],
            kwargs={k: wrap_lineage_id(v) for k, v in self.kwargs.items()},
            result_id=wrap_lineage_id(self.result_id),
            action_type=self.action_type,
            user_code_id=self.user_code_id,
        )


class ActionObjectDB(CommonMixin, Base):
    id: Mapped[UID] = mapped_column(
        UIDTypeDecorator, primary_key=True, default=uuid.uuid4
    )
    syft_blob_storage_entry_id: Mapped[UID | None] = mapped_column(
        UIDTypeDecorator, default=None
    )
    syft_action_data_cache: Mapped[bytes | None] = mapped_column(default=None)
    syft_history_hash: Mapped[int | None] = mapped_column(default=None)
    syft_server_uid: Mapped[UID | None] = mapped_column(UIDTypeDecorator, default=None)
    syft_twin_type: Mapped[TwinMode] = mapped_column()
    syft_action_data_type: Mapped[str | None] = mapped_column(default=None)
    syft_action_data_repr_: Mapped[str | None] = mapped_column(default=None)
    syft_action_data_str_: Mapped[str | None] = mapped_column(default=None)
    syft_has_bool_attr: Mapped[bool] = mapped_column(default=None)
    syft_resolved: Mapped[bool] = mapped_column(default=True)
    syft_action_data_server_id: Mapped[UID | None] = mapped_column(
        UIDTypeDecorator, default=None
    )
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
            id=obj.id,
            syft_blob_storage_entry_id=obj.syft_blob_storage_entry_id,
            # syft_action_data_cache=sy.serialize(
            #     obj.syft_action_data_cache, to_bytes=True
            # )
            # if obj.syft_action_data_cache is not None
            # else None,
            syft_history_hash=obj.syft_history_hash,
            syft_server_uid=obj.syft_server_uid,
            syft_twin_type=obj.syft_twin_type,
            syft_action_data_type=action_data_type,
            syft_action_data_repr_=obj.syft_action_data_repr_,
            syft_action_data_str_=obj.syft_action_data_str_,
            syft_has_bool_attr=obj.syft_has_bool_attr,
            syft_resolved=obj.syft_resolved,
            created_at=obj.syft_created_at,
            syft_action_data_server_id=obj.syft_action_data_server_id
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
            id=self.id,
            syft_blob_storage_entry_id=self.syft_blob_storage_entry_id,
            syft_action_data_cache=sy.deserialize(
                self.syft_action_data_cache, from_bytes=True
            )
            if self.syft_action_data_cache is not None
            else None,
            syft_history_hash=self.syft_history_hash,
            syft_server_uid=self.syft_server_uid,
            syft_twin_type=self.syft_twin_type,
            syft_action_data_type=dtype,
            syft_action_data_repr_=self.syft_action_data_repr_,
            syft_action_data_str_=self.syft_action_data_str_,
            syft_has_bool_attr=self.syft_has_bool_attr,
            syft_resolved=self.syft_resolved,
            created_at=self.created_at if self.created_at else None,
            syft_action_data_server_id=self.syft_action_data_server_id,
            syft_action_saved_to_blob_store=self.syft_action_saved_to_blob_store,
        )
