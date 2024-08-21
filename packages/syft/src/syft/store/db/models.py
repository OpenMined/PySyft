# stdlib

# third party
import sqlalchemy as sa
from sqlalchemy import TypeDecorator
from sqlalchemy.orm import DeclarativeBase

# relative
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
