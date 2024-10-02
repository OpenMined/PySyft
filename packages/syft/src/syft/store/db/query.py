# stdlib
from abc import ABC
from abc import abstractmethod
import enum
from typing import Any
from typing import Literal

# third party
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Dialect
from sqlalchemy import Result
from sqlalchemy import Select
from sqlalchemy import Table
from sqlalchemy import func
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session
from typing_extensions import Self

# relative
from ...serde.json_serde import serialize_json
from ...server.credentials import SyftVerifyKey
from ...service.action.action_permissions import ActionObjectPermission
from ...service.action.action_permissions import ActionPermission
from ...service.user.user_roles import ServiceRole
from ...types.syft_object import SyftObject
from ...types.uid import UID
from .errors import StashDBException
from .schema import PostgresBase
from .schema import SQLiteBase


class FilterOperator(enum.Enum):
    EQ = "eq"
    CONTAINS = "contains"


class Query(ABC):
    def __init__(self, object_type: type[SyftObject]) -> None:
        self.object_type: type = object_type
        self.table: Table = self._get_table(object_type)
        self.stmt: Select = self.table.select()

    @abstractmethod
    def _get_table(self, object_type: type[SyftObject]) -> Table:
        raise NotImplementedError

    @staticmethod
    def get_query_class(dialect: str | Dialect) -> "type[Query]":
        if isinstance(dialect, Dialect):
            dialect = dialect.name

        if dialect == "sqlite":
            return SQLiteQuery
        elif dialect == "postgresql":
            return PostgresQuery
        else:
            raise ValueError(f"Unsupported dialect {dialect}")

    @classmethod
    def create(cls, object_type: type[SyftObject], dialect: str | Dialect) -> "Query":
        """Create a query object for the given object type and dialect."""
        query_class = cls.get_query_class(dialect)
        return query_class(object_type)

    def execute(self, session: Session) -> Result:
        """Execute the query using the given session."""
        try:
            return session.execute(self.stmt)
        except DatabaseError as e:
            raise StashDBException.from_sqlalchemy_error(e) from e

    def with_permissions(
        self,
        credentials: SyftVerifyKey,
        role: ServiceRole,
        permission: ActionPermission = ActionPermission.READ,
    ) -> Self:
        """Add a permission check to the query.

        If the user has a role below DATA_OWNER, the query will be filtered to only include objects
        that the user has the specified permission on.

        Args:
            credentials (SyftVerifyKey): user verify key
            role (ServiceRole): role of the user
            permission (ActionPermission, optional): Type of permission to check for.
                Defaults to ActionPermission.READ.

        Returns:
            Self: The query object with the permission check applied
        """
        if role in (ServiceRole.ADMIN, ServiceRole.DATA_OWNER):
            return self

        ao_permission = ActionObjectPermission(
            uid=UID(),  # dummy uid, we just need the permission string
            credentials=credentials,
            permission=permission,
        )

        permission_clause = self._make_permissions_clause(ao_permission)
        self.stmt = self.stmt.where(permission_clause)

        return self

    def filter(self, field: str, operator: str | FilterOperator, value: Any) -> Self:
        """Add a filter to the query.

        example usage:
        Query(User).filter("name", "eq", "Alice")
        Query(User).filter("friends", "contains", "Bob")

        Args:
            field (str): Field to filter on
            operator (str): Operator to use for the filter
            value (Any): Value to filter on

        Raises:
            ValueError: If the operator is not supported

        Returns:
            Self: The query object with the filter applied
        """
        filter = self._create_filter_clause(self.table, field, operator, value)
        self.stmt = self.stmt.where(filter)
        return self

    def filter_and(self, *filters: tuple[str, str | FilterOperator, Any]) -> Self:
        """Add filters to the query using an AND clause.

        example usage:
        Query(User).filter_and(
            ("name", "eq", "Alice"),
            ("age", "eq", 30),
        )

        Args:
            field (str): Field to filter on
            operator (str): Operator to use for the filter
            value (Any): Value to filter on

        Raises:
            ValueError: If the operator is not supported

        Returns:
            Self: The query object with the filter applied
        """
        filter_clauses = [
            self._create_filter_clause(self.table, field, operator, value)
            for field, operator, value in filters
        ]

        self.stmt = self.stmt.where(sa.and_(*filter_clauses))
        return self

    def filter_or(self, *filters: tuple[str, str | FilterOperator, Any]) -> Self:
        """Add filters to the query using an OR clause.

        example usage:
        Query(User).filter_or(
            ("name", "eq", "Alice"),
            ("age", "eq", 30),
        )

        Args:
            field (str): Field to filter on
            operator (str): Operator to use for the filter
            value (Any): Value to filter on

        Raises:
            ValueError: If the operator is not supported

        Returns:
            Self: The query object with the filter applied
        """
        filter_clauses = [
            self._create_filter_clause(self.table, field, operator, value)
            for field, operator, value in filters
        ]

        self.stmt = self.stmt.where(sa.or_(*filter_clauses))
        return self

    def _create_filter_clause(
        self,
        table: Table,
        field: str,
        operator: str | FilterOperator,
        value: Any,
    ) -> sa.sql.elements.BinaryExpression:
        if isinstance(operator, str):
            try:
                operator = FilterOperator(operator.lower())
            except ValueError:
                raise ValueError(f"Filter operator {operator} not supported")

        if operator == FilterOperator.EQ:
            return self._eq_filter(table, field, value)
        elif operator == FilterOperator.CONTAINS:
            return self._contains_filter(table, field, value)

    def order_by(
        self,
        field: str | None = None,
        order: Literal["asc", "desc"] | None = None,
    ) -> Self:
        """Add an order by clause to the query, with sensible defaults if field or order is not provided.

        Args:
            field (Optional[str]): field to order by. If None, uses the default field.
            order (Optional[Literal["asc", "desc"]]): Order to use ("asc" or "desc").
                Defaults to 'asc' if field is provided and order is not, or the default order otherwise.

        Raises:
            ValueError: If the order is not "asc" or "desc"

        Returns:
            Self: The query object with the order by clause applied.
        """
        # Determine the field and order defaults if not provided
        if field is None:
            if hasattr(self.object_type, "__order_by__"):
                default_field, default_order = self.object_type.__order_by__
            else:
                default_field, default_order = "_created_at", "desc"
            field = default_field
        else:
            # If field is provided but order is not, default to 'asc'
            default_order = "asc"
        order = order or default_order

        column = self._get_column(field)

        if isinstance(column.type, sa.JSON):
            column = sa.cast(column, sa.String)

        if order.lower() == "asc":
            self.stmt = self.stmt.order_by(column.asc())

        elif order.lower() == "desc":
            self.stmt = self.stmt.order_by(column.desc())
        else:
            raise ValueError(f"Invalid sort order {order}")

        return self

    def limit(self, limit: int | None) -> Self:
        """Add a limit clause to the query."""
        if limit is None:
            return self

        if limit < 0:
            raise ValueError("Limit must be a positive integer")
        self.stmt = self.stmt.limit(limit)

        return self

    def offset(self, offset: int) -> Self:
        """Add an offset clause to the query."""
        if offset < 0:
            raise ValueError("Offset must be a positive integer")

        self.stmt = self.stmt.offset(offset)
        return self

    @abstractmethod
    def _make_permissions_clause(
        self,
        permission: ActionObjectPermission,
    ) -> sa.sql.elements.BinaryExpression:
        pass

    @abstractmethod
    def _contains_filter(
        self,
        table: Table,
        field: str,
        value: Any,
    ) -> sa.sql.elements.BinaryExpression:
        pass

    def _get_column(self, column: str) -> Column:
        if column == "id":
            return self.table.c.id
        if column == "created_date" or column == "_created_at":
            return self.table.c._created_at
        elif column == "updated_date" or column == "_updated_at":
            return self.table.c._updated_at
        elif column == "deleted_date" or column == "_deleted_at":
            return self.table.c._deleted_at

        return self.table.c.fields[column]


class SQLiteQuery(Query):
    def _make_permissions_clause(
        self,
        permission: ActionObjectPermission,
    ) -> sa.sql.elements.BinaryExpression:
        permission_string = permission.permission_string
        compound_permission_string = permission.compound_permission_string
        return sa.or_(
            self.table.c.permissions.contains(permission_string),
            self.table.c.permissions.contains(compound_permission_string),
        )

    def _get_table(self, object_type: type[SyftObject]) -> Table:
        cname = object_type.__canonical_name__
        if cname not in SQLiteBase.metadata.tables:
            raise ValueError(f"Table for {cname} not found")
        return SQLiteBase.metadata.tables[cname]

    def _contains_filter(
        self,
        table: Table,
        field: str,
        value: Any,
    ) -> sa.sql.elements.BinaryExpression:
        field_value = serialize_json(value)
        return table.c.fields[field].contains(func.json_quote(field_value))

    def _eq_filter(
        self,
        table: Table,
        field: str,
        value: Any,
    ) -> sa.sql.elements.BinaryExpression:
        if field == "id":
            return table.c.id == UID(value)

        if "." in field:
            # magic!
            field = field.split(".")  # type: ignore

        json_value = serialize_json(value)
        return table.c.fields[field] == func.json_quote(json_value)


class PostgresQuery(Query):
    def _make_permissions_clause(
        self, permission: ActionObjectPermission
    ) -> sa.sql.elements.BinaryExpression:
        permission_string = [permission.permission_string]
        compound_permission_string = [permission.compound_permission_string]
        return sa.or_(
            self.table.c.permissions.contains(permission_string),
            self.table.c.permissions.contains(compound_permission_string),
        )

    def _contains_filter(
        self,
        table: Table,
        field: str,
        value: Any,
    ) -> sa.sql.elements.BinaryExpression:
        field_value = serialize_json(value)
        col = sa.cast(table.c.fields[field], sa.Text)
        val = sa.cast(field_value, sa.Text)
        return col.contains(val)

    def _get_table(self, object_type: type[SyftObject]) -> Table:
        cname = object_type.__canonical_name__
        if cname not in PostgresBase.metadata.tables:
            raise ValueError(f"Table for {cname} not found")
        return PostgresBase.metadata.tables[cname]

    def _eq_filter(
        self,
        table: Table,
        field: str,
        value: Any,
    ) -> sa.sql.elements.BinaryExpression:
        if field == "id":
            return table.c.id == UID(value)

        if "." in field:
            # magic!
            field = field.split(".")  # type: ignore

        json_value = serialize_json(value)
        # NOTE: there might be a bug with casting everything to text
        return table.c.fields[field].astext == sa.cast(json_value, sa.Text)
