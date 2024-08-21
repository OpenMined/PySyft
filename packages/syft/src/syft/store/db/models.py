# stdlib
import uuid

# third party
import sqlalchemy as sa
from sqlalchemy import TypeDecorator
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.types import JSON

# relative
from ...types.datetime import DateTime
from ...types.uid import UID


class Base(DeclarativeBase):
    pass


class UIDTypeDecorator(TypeDecorator):
    """Converts between Syft UID and UUID."""

    impl = sa.UUID
    cache_ok = True

    def process_bind_param(self, value, dialect):  # type: ignore
        if value is not None:
            return value.value

    def process_result_value(self, value, dialect):  # type: ignore
        if value is not None:
            return UID(value)


class CommonMixin:
    id: Mapped[UID] = mapped_column(
        default=uuid.uuid4,
        primary_key=True,
    )
    created_at: Mapped[DateTime] = mapped_column(server_default=sa.func.now())

    updated_at: Mapped[DateTime] = mapped_column(
        server_default=sa.func.now(),
        server_onupdate=sa.func.now(),
    )
    json_document: Mapped[dict] = mapped_column(JSON, default={})
