# stdlib
from datetime import datetime
import uuid

# third party
import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

# relative
from .job_stash import JobStatus
from .job_stash import JobType

engine = create_engine("sqlite://")


class Base(DeclarativeBase):
    pass


class CommonMixin:
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
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


class Job(CommonMixin, Base):
    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    server_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    result_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    resolved: Mapped[bool] = mapped_column(default=False)
    status: Mapped[JobStatus] = mapped_column(default=JobStatus.CREATED)
    n_iters: Mapped[int | None] = mapped_column(default=None)
    current_iter: Mapped[int | None] = mapped_column(default=None)
    action: Mapped[bytes | None] = mapped_column(default=None)

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("job.id"), default=None
    )
    children: Mapped[list["Job"]] = relationship(
        "Job",
        back_populates="parent",
        remote_side=[id],
    )
    parent: Mapped["Job"] = relationship(
        back_populates="children",
        # foreign_keys=[parent_id],
        # remote_side=[id],
    )

    job_pid: Mapped[int | None] = mapped_column(default=None)
    job_worker_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    log_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    user_code_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    requested_by: Mapped[uuid.UUID | None] = mapped_column(default=None)
    job_type: Mapped[JobType] = mapped_column(default=JobType.JOB)
