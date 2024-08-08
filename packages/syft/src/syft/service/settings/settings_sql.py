# stdlib

# third party
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from typing_extensions import Self

# syft absolute
import syft as sy

# relative
from ...abstract_server import ServerSideType
from ...abstract_server import ServerType
from ...server.credentials import SyftVerifyKey
from ..job.job_sql import Base
from ..job.job_sql import CommonMixin
from ..job.job_sql import PermissionMixin
from ..job.job_sql import VerifyKeyTypeDecorator
from .settings import ServerSettings


class ServerSettingsDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "server_settings"
    name: Mapped[str]
    deployed_on: Mapped[str]
    organization: Mapped[str]
    verify_key: Mapped[SyftVerifyKey] = mapped_column(VerifyKeyTypeDecorator)
    on_board: Mapped[bool]
    description: Mapped[str]
    server_type: Mapped[ServerType]
    signup_enabled: Mapped[bool]
    admin_email: Mapped[str]
    server_side_type: Mapped[ServerSideType]
    show_warnings: Mapped[bool]
    association_request_auto_approval: Mapped[bool]
    eager_execution_enabled: Mapped[bool]
    default_worker_pool: Mapped[str]
    welcome_markdown: Mapped[bytes]
    notifications_enabled: Mapped[bool]

    @classmethod
    def from_obj(cls, obj: ServerSettings) -> Self:
        custom_mapping = {
            "created_date": "created_at",
            "updated_date": "updated_at",
            "deleted_date": "deleted_at",
        }
        custom_converters = {
            "welcome_markdown": lambda x: sy.serialize(x, to_bytes=True) if x else None,
        }
        exclude = {"syft_server_location", "syft_client_verify_key"}
        kwargs = {}
        for key, value in obj.__dict__.items():
            if key in exclude:
                continue
            if key in custom_converters:
                value = custom_converters[key](value)
            if key in custom_mapping:
                key = custom_mapping[key]
            kwargs[key] = value
        return cls(**kwargs)

    def to_obj(self) -> ServerSettings:
        custom_mapping = {
            "created_at": "created_date",
            "updated_at": "updated_date",
            "deleted_at": "deleted_date",
        }
        custom_converters = {
            "welcome_markdown": lambda x: sy.deserialize(x, from_bytes=True)
            if x
            else None,
        }
        kwargs = {}
        for key in ServerSettings.__annotations__:
            value = getattr(self, key)

            if key in custom_converters:
                value = custom_converters[key](value)
            if key in custom_mapping:
                key = custom_mapping[key]
            kwargs[key] = value
        return ServerSettings(**kwargs)


ServerSettingsDB._init_perms()
