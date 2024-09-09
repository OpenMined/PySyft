# stdlib
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Literal

# third party
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Dialect
from sqlalchemy import Result
from sqlalchemy import Select
from sqlalchemy import Table
from sqlalchemy import dialects
from sqlalchemy import func
from sqlalchemy import select
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
from .sqlite_db import OBJECT_TYPE_TO_TABLE


class Query(ABC):
    dialect: Dialect

    def __init__(self, object_type: type[SyftObject]) -> None:
        self.object_type: type = object_type
        self.table: Table = OBJECT_TYPE_TO_TABLE[object_type]
        self.stmt: Select = select([self.table])

    def compile(self) -> str:
        """
        Compile the query to a string, for debugging purposes.
        """
        return self.stmt.compile(
            compile_kwargs={"literal_binds": True},
            dialect=self.dialect,
        )

    def execute(self, session: Session) -> Result:
        """Execute the query using the given session."""
        return session.execute(self.stmt)

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
        if role in (ServiceRole.ADMIN, ServiceRole.SUPER_ADMIN):
            return self

        permission = ActionObjectPermission(
            uid=UID(),  # dummy uid, we just need the permission string
            credentials=credentials,
            permission=permission,
        )

        permission_clause = self._make_permissions_clause(permission)
        self.stmt = self.stmt.where(permission_clause)

        return self

    def filter(self, field: str, operator: str, value: Any) -> Self:
        """Add a filter to the query.

        example usage:
        Query(User).filter("name", "==", "Alice")
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
        if operator not in {"==", "!=", "contains"}:
            raise ValueError(f"Operation {operator} not supported")

        if operator == "==":
            filter = self._eq_filter(self.table, field, value)
            self.stmt = self.stmt.where(filter)
        elif operator == "contains":
            filter = self._contains_filter(self.table, field, value)
            self.stmt = self.stmt.where(filter)

        return self

    def order_by(self, field: str, order: Literal["asc", "desc"] = "asc") -> Self:
        """Add an order by clause to the query.

        Args:
            field (str): field to order by.
            order (Literal["asc", "desc"], optional): Order to use.
                Defaults to "asc".

        Raises:
            ValueError: If the order is not "asc" or "desc"

        Returns:
            Self: The query object with the order by clause applied
        """
        column = self._get_column(field)

        if order.lower() == "asc":
            self.stmt = self.stmt.order_by(column)
        elif order.lower() == "desc":
            self.stmt = self.stmt.order_by(column.desc())
        else:
            raise ValueError(f"Invalid sort order {order}")  # type: ignore

        return self

    def limit(self, limit: int) -> Self:
        """Add a limit clause to the query."""
        self.stmt = self.stmt.limit(limit)
        return self

    def offset(self, offset: int) -> Self:
        """Add an offset clause to the query."""
        self.stmt = self.stmt.offset(offset)
        return self

    @abstractmethod
    def _make_permissions_clause(
        self,
        permission: ActionObjectPermission,
    ) -> sa.sql.elements.BinaryExpression:
        pass

    def default_order(self) -> Self:
        if hasattr(self.object_type, "__order_by__"):
            field, order = self.object_type.__order_by__
        else:
            field, order = "_created_at", "desc"

        return self.order_by(field, order)

    def _eq_filter(
        self,
        table: Table,
        field: str,
        value: Any,
    ) -> sa.sql.elements.BinaryExpression:
        if field == "id":
            return table.c.id == UID(value)

        json_value = serialize_json(value)
        return table.c.fields[field] == func.json_quote(json_value)

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
    dialect = dialects.sqlite.dialect

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

    def _contains_filter(
        self,
        table: Table,
        field: str,
        value: Any,
    ) -> sa.sql.elements.BinaryExpression:
        field_value = serialize_json(value)
        return table.c.fields[field].contains(func.json_quote(field_value))


class PostgresQuery(Query):
    dialect = dialects.postgresql.dialect

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
        field_value = [serialize_json(value)]
        return table.c.fields[field].contains(field_value)
