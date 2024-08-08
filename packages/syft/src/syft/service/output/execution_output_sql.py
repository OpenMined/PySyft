# stdlib
from typing import TYPE_CHECKING
from syft.service.output.output_service import ExecutionOutput
from typing_extensions import Self

# third party
import sqlalchemy as sa
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

# syft absolute
import syft as sy

# relative
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID
from ..job.job_sql import Base, VerifyKeyTypeDecorator
from ..job.job_sql import CommonMixin
from ..job.job_sql import PermissionMixin
from ..job.job_sql import UIDTypeDecorator

if TYPE_CHECKING:
    from ..job.job_sql import JobDB
    from syft.service.code.user_code_sql import UserCodeDB


class ExecutionOutputDB(CommonMixin, Base, PermissionMixin):  # noqa: F821
    __tablename__ = "execution_outputs"

    executing_user_verify_key: Mapped[SyftVerifyKey] = mapped_column(
        VerifyKeyTypeDecorator
    )
    output_ids: Mapped[bytes]
    input_ids: Mapped[bytes]

    user_code_link: Mapped[bytes]

    user_code_id: Mapped[UID] = mapped_column(
        UIDTypeDecorator, sa.ForeignKey("user_codes.id")
    )
    user_code: Mapped["UserCodeDB"] = relationship(
        back_populates="execution_output", uselist=False
    )

    job_id: Mapped[UID | None] = mapped_column(
        UIDTypeDecorator, sa.ForeignKey("jobs.id")
    )
    job: Mapped["JobDB"] = relationship(
        back_populates="execution_output", uselist=False
    )

    @classmethod
    def from_obj(cls, obj: ExecutionOutput) -> Self:
        return cls(
            id=obj.id,
            executing_user_verify_key=obj.executing_user_verify_key,
            user_code_id=obj.user_code_id,
            output_ids=sy.serialize(obj.output_ids, to_bytes=True),
            input_ids=sy.serialize(obj.input_ids, to_bytes=True),
            job_id=obj.job_id,
            user_code_link=sy.serialize(obj.user_code_link, to_bytes=True),
        )

    def to_obj(self) -> ExecutionOutput:
        return ExecutionOutput(
            id=self.id,
            executing_user_verify_key=self.executing_user_verify_key,
            user_code_id=self.user_code_id,
            output_ids=sy.deserialize(self.output_ids, from_bytes=True),
            input_ids=sy.deserialize(self.input_ids, from_bytes=True),
            job_id=self.job_id,
            user_code_link=sy.deserialize(self.user_code_link, from_bytes=True),
        )


ExecutionOutputDB._init_perms()
