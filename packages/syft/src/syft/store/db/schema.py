# stdlib

# stdlib
import uuid

# third party
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Dialect
from sqlalchemy import Table
from sqlalchemy import TypeDecorator
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import JSON

# relative
from ...types.syft_object import SyftObject
from ...types.uid import UID


class SQLiteBase(DeclarativeBase):
    pass


class PostgresBase(DeclarativeBase):
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


def create_table(
    object_type: type[SyftObject],
    dialect: Dialect,
) -> Table:
    """Create a table for a given SYftObject type, and add it to the metadata.

    To create the table on the database, you must call `Base.metadata.create_all(engine)`.

    Args:
        object_type (type[SyftObject]): The type of the object to create a table for.
        dialect (Dialect): The dialect of the database.

    Returns:
        Table: The created table.
    """
    table_name = object_type.__canonical_name__
    dialect_name = dialect.name

    fields_type = JSON if dialect_name == "sqlite" else postgresql.JSON
    permissions_type = JSON if dialect_name == "sqlite" else postgresql.JSONB
    storage_permissions_type = JSON if dialect_name == "sqlite" else postgresql.JSONB

    Base = SQLiteBase if dialect_name == "sqlite" else PostgresBase

    if table_name not in Base.metadata.tables:
        Table(
            object_type.__canonical_name__,
            Base.metadata,
            Column("id", UIDTypeDecorator, primary_key=True, default=uuid.uuid4),
            Column("fields", fields_type, default={}),
            Column("permissions", permissions_type, default=[]),
            Column(
                "storage_permissions",
                storage_permissions_type,
                default=[],
            ),
            Column(
                "_created_at", sa.DateTime, server_default=sa.func.now(), index=True
            ),
            Column("_updated_at", sa.DateTime, server_onupdate=sa.func.now()),
            Column("_deleted_at", sa.DateTime, index=True),
        )

    return Base.metadata.tables[table_name]
