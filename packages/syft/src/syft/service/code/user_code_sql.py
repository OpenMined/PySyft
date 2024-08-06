# stdlib
from datetime import datetime
import uuid

# third party
import sqlalchemy as sa
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from syft.abstract_server import ServerSideType
from syft.server.credentials import SyftVerifyKey
from syft.service.action.action_permissions import ActionPermission
from syft.service.code.user_code import UserCode
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


class UserCodeDB(CommonMixin, Base, PermissionMixin):  # noqa: F821
    __tablename__ = "user_codes"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    server_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    user_verify_key: Mapped[str]
    raw_code: Mapped[str]
    input_policy_type: Mapped[bytes]
    input_policy_init_kwargs: Mapped[bytes]
    input_policy_state: Mapped[bytes]
    output_policy_type: Mapped[bytes]
    output_policy_init_kwargs: Mapped[bytes]
    output_policy_state: Mapped[bytes]
    parsed_code: Mapped[str]
    service_func_name: Mapped[str]
    unique_func_name: Mapped[str]
    user_unique_func_name: Mapped[str]
    code_hash: Mapped[str]
    signature: Mapped[bytes]
    status_link: Mapped[bytes]
    input_kwargs: Mapped[list[str]] = mapped_column(sa.JSON, default=[])
    submit_time: Mapped[datetime | None] = mapped_column(default=None)
    uses_datasite: Mapped[bool] = mapped_column(default=False)
    worker_pool_name: Mapped[str | None] = mapped_column(default=None)
    origin_server_side_type: Mapped[ServerSideType]
    l0_deny_reason: Mapped[str | None] = mapped_column(default=None)
    nested_codes: Mapped[bytes] = mapped_column(sa.LargeBinary, default=b"")
    status_collection = relationship(
        "UserCodeStatusCollectionDB", back_populates="user_code"
    )

    @classmethod
    def from_obj(cls, obj: UserCode) -> "UserCodeDB":
        return cls(
            id=unwrap_uid(obj.id),
            server_uid=unwrap_uid(obj.server_uid),
            user_verify_key=str(obj.user_verify_key),
            raw_code=obj.raw_code,
            input_policy_type=sy.serialize(obj.input_policy_type, to_bytes=True),
            input_policy_init_kwargs=sy.serialize(
                obj.input_policy_init_kwargs, to_bytes=True
            ),
            input_policy_state=obj.input_policy_state,
            output_policy_type=sy.serialize(obj.output_policy_type, to_bytes=True),
            output_policy_init_kwargs=sy.serialize(
                obj.output_policy_init_kwargs, to_bytes=True
            ),
            output_policy_state=obj.output_policy_state,
            parsed_code=obj.parsed_code,
            service_func_name=obj.service_func_name,
            unique_func_name=obj.unique_func_name,
            user_unique_func_name=obj.user_unique_func_name,
            code_hash=obj.code_hash,
            signature=sy.serialize(obj.signature, to_bytes=True),
            status_link=sy.serialize(obj.status_link, to_bytes=True),
            input_kwargs=obj.input_kwargs,
            submit_time=obj.submit_time.to_datetime() if obj.submit_time else None,
            uses_datasite=obj.uses_datasite,
            worker_pool_name=obj.worker_pool_name,
            origin_server_side_type=obj.origin_server_side_type,
            l0_deny_reason=obj.l0_deny_reason,
            nested_codes=sy.serialize(obj.nested_codes, to_bytes=True)
            if obj.nested_codes
            else b"",
        )

    def to_obj(self) -> UserCode:
        return UserCode(
            id=wrap_uid(self.id),
            server_uid=wrap_uid(self.server_uid),
            user_verify_key=SyftVerifyKey(self.user_verify_key),
            raw_code=self.raw_code,
            input_policy_type=sy.deserialize(self.input_policy_type, from_bytes=True),
            input_policy_init_kwargs=sy.deserialize(
                self.input_policy_init_kwargs, from_bytes=True
            ),
            input_policy_state=self.input_policy_state,
            output_policy_type=sy.deserialize(self.output_policy_type, from_bytes=True),
            output_policy_init_kwargs=sy.deserialize(
                self.output_policy_init_kwargs, from_bytes=True
            ),
            output_policy_state=self.output_policy_state,
            parsed_code=self.parsed_code,
            service_func_name=self.service_func_name,
            unique_func_name=self.unique_func_name,
            user_unique_func_name=self.user_unique_func_name,
            code_hash=self.code_hash,
            signature=sy.deserialize(self.signature, from_bytes=True),
            status_link=sy.deserialize(self.status_link, from_bytes=True),
            input_kwargs=self.input_kwargs,
            submit_time=DateTime.from_datetime(self.submit_time)
            if self.submit_time
            else None,
            uses_datasite=self.uses_datasite,
            worker_pool_name=self.worker_pool_name,
            origin_server_side_type=self.origin_server_side_type,
            l0_deny_reason=self.l0_deny_reason,
            nested_codes=sy.deserialize(self.nested_codes, from_bytes=True)
            if self.nested_codes
            else None,
        )


UserCodeDB._init_perms()
