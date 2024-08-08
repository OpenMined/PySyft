# stdlib

# third party
from typing import TYPE_CHECKING
from colorama import Fore
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column, relationship
from syft.service.log.log import SyftLog
from syft.types.uid import UID
from typing_extensions import Self

# syft absolute
import syft as sy

# relative
from ..job.job_sql import Base, UIDTypeDecorator
from ..job.job_sql import CommonMixin
from ..job.job_sql import PermissionMixin

if TYPE_CHECKING:
    from ..job.job_sql import JobDB


class SyftLogDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "syft_logs"
    stdout: Mapped[str]
    stderr: Mapped[str]
    job_id: Mapped[UID] = mapped_column(UIDTypeDecorator, ForeignKey("jobs.id"))
    job: Mapped["JobDB"] = relationship(back_populates="log", uselist=False)

    @classmethod
    def from_obj(cls, obj: SyftLog) -> Self:
        return cls(
            id=obj.id,
            stdout=obj.stdout,
            stderr=obj.stderr,
            job_id=obj.job_id,
        )

    def to_obj(self) -> SyftLog:
        return SyftLog(
            id=self.id,
            stdout=self.stdout,
            stderr=self.stderr,
            job_id=self.job.id,
        )


SyftLogDB._init_perms()
