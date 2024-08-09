# stdlib
from typing import TYPE_CHECKING

# third party
import sqlalchemy as sa
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

# syft absolute
import syft as sy
from syft.service.code_history.code_history import CodeHistory

# relative
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID
from ..job.base_sql import Base, VerifyKeyTypeDecorator
from ..job.base_sql import CommonMixin
from ..job.base_sql import PermissionMixin
from ..job.base_sql import UIDTypeDecorator

if TYPE_CHECKING:
    from ..job.job_sql import JobDB
    from syft.service.output.execution_output_sql import ExecutionOutputDB


class CodeHistoryDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "code_histories"

    server_uid: Mapped[UID] = mapped_column(UIDTypeDecorator)
    user_verify_key: Mapped[SyftVerifyKey] = mapped_column(VerifyKeyTypeDecorator)
    service_func_name: Mapped[str]
    comment_history: Mapped[list[str]] = mapped_column(sa.JSON, default=[])
    user_code_history: Mapped[bytes]

    @classmethod
    def from_obj(cls, obj: CodeHistory) -> "CodeHistoryDB":
        return cls(
            id=obj.id,
            server_uid=obj.server_uid,
            user_verify_key=obj.user_verify_key,
            service_func_name=obj.service_func_name,
            comment_history=obj.comment_history,
            user_code_history=sy.serialize(obj.user_code_history, to_bytes=True),
        )

    def to_obj(self) -> CodeHistory:
        return CodeHistory(
            id=self.id,
            server_uid=self.server_uid,
            user_verify_key=self.user_verify_key,
            service_func_name=self.service_func_name,
            comment_history=self.comment_history,
            user_code_history=sy.deserialize(self.user_code_history, from_bytes=True),
        )


CodeHistoryDB._init_perms()
