# stdlib

# stdlib
from typing import Any
from typing import Generic
import uuid

# third party
from result import Err
from result import Ok
from result import Result
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Row
from sqlalchemy import Select
from sqlalchemy import Table
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.types import JSON
from typing_extensions import TypeVar

# relative
from ...serde.json_serde import deserialize_json
from ...serde.json_serde import serialize_json
from ...server.credentials import SyftVerifyKey
from ...service.action.action_permissions import ActionObjectEXECUTE
from ...service.action.action_permissions import ActionObjectOWNER
from ...service.action.action_permissions import ActionObjectPermission
from ...service.action.action_permissions import ActionObjectREAD
from ...service.action.action_permissions import ActionObjectWRITE
from ...service.action.action_permissions import ActionPermission
from ...service.action.action_permissions import StoragePermission
from ...service.response import SyftSuccess
from ...service.user.user_roles import ServiceRole
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..document_store import DocumentStore
from .models import Base
from .models import UIDTypeDecorator
from .sqlite_db import SQLiteDBManager

SyftT = TypeVar("SyftT", bound=SyftObject)
T = TypeVar("T")


class ObjectStash(Generic[SyftT]):
    object_type: type[SyftT]

    def __init__(self, store: DocumentStore) -> None:
        self.server_uid = store.server_uid
        self.root_verify_key = store.root_verify_key
        # is there a better way to init the table
        _ = self.table
        self.db = SQLiteDBManager(self.server_uid)

    def check_type(self, obj: T, type_: type) -> Result[T, str]:
        return (
            Ok(obj)
            if isinstance(obj, type_)
            else Err(f"{type(obj)} does not match required type: {type_}")
        )

    @property
    def session(self) -> Session:
        return self.db.session

    @property
    def table(self) -> Table:
        # need to call Base.metadata.create_all(engine) to create the table
        table_name = self.object_type.__canonical_name__
        if table_name not in Base.metadata.tables:
            Table(
                self.object_type.__canonical_name__,
                Base.metadata,
                Column("id", UIDTypeDecorator, primary_key=True, default=uuid.uuid4),
                Column("fields", JSON, default={}),
                Column("permissions", JSON, default=[]),
                Column(
                    "created_at", sa.DateTime, server_default=sa.func.now(), index=True
                ),
                Column("updated_at", sa.DateTime, server_onupdate=sa.func.now()),
            )
        return Base.metadata.tables[table_name]

    def _print_query(self, stmt: sa.sql.select) -> None:
        print(
            stmt.compile(
                compile_kwargs={"literal_binds": True},
                dialect=self.session.bind.dialect,
            )
        )

    def is_unique(self, obj: SyftT) -> bool:
        unique_fields = self.object_type.__attr_unique__
        if not unique_fields:
            return True
        filters = []
        for filter_name in unique_fields:
            field_value = getattr(obj, filter_name, None)
            if field_value is None:
                continue
            filt = self._get_field_filter(
                field_name=filter_name,
                # is the str cast correct? how to handle SyftVerifyKey?
                field_value=str(field_value),
            )
            filters.append(filt)

        stmt = self.table.select().where(sa.or_(*filters))
        results = self.session.execute(stmt).all()
        if len(results) > 1:
            return False
        elif len(results) == 1:
            result = results[0]
            return result.id == obj.id
        return True

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[SyftT | None, str]:
        result = self.session.execute(
            self.table.select().where(
                sa.and_(
                    self._get_field_filter("id", uid),
                    self._get_permission_filter(credentials),
                )
            )
        ).first()
        if result is None:
            return Ok(None)
        return Ok(self.row_as_obj(result))

    def _get_field_filter(
        self,
        field_name: str,
        field_value: str,
        table: Table | None = None,
    ) -> sa.sql.elements.BinaryExpression:
        table = table if table is not None else self.table
        if field_name == "id":
            return table.c.id == field_value
        return table.c.fields[field_name] == func.json_quote(field_value)

    def _get_by_fields(
        self,
        credentials: SyftVerifyKey,
        fields: dict[str, str],
        table: Table | None = None,
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
    ) -> Result[Row, str]:
        table = table if table is not None else self.table
        filters = []
        for field_name, field_value in fields.items():
            filt = self._get_field_filter(field_name, field_value, table=table)
            filters.append(filt)

        stmt = table.select().where(
            sa.and_(
                sa.and_(*filters),
                self._get_permission_filter(credentials),
            )
        )
        stmt = self._apply_order_by(stmt, order_by, sort_order)
        stmt = self._apply_limit_offset(stmt, limit, offset)

        result = self.session.execute(stmt)
        return result

    def get_one_by_field(
        self, credentials: SyftVerifyKey, field_name: str, field_value: str
    ) -> Result[SyftT | None, str]:
        result = self._get_by_fields(
            credentials=credentials,
            fields={field_name: field_value},
        ).first()
        if result is None:
            return Ok(None)
        return Ok(self.row_as_obj(result))

    def get_one_by_fields(
        self,
        credentials: SyftVerifyKey,
        fields: dict[str, str],
    ) -> Result[SyftT | None, str]:
        result = self._get_by_fields(
            credentials=credentials,
            fields=fields,
        ).first()
        if result is None:
            return Ok(None)
        return Ok(self.row_as_obj(result))

    def get_all_by_fields(
        self,
        credentials: SyftVerifyKey,
        fields: dict[str, str],
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
    ) -> Result[list[SyftT], str]:
        # sanity check if the field is not a list, set etc.
        for field_name in fields:
            if field_name not in self.object_type.__annotations__:
                return Err(
                    f"Field {field_name} not found in {self.object_type.__name__}"
                )

        result = self._get_by_fields(
            credentials=credentials,
            fields=fields,
            order_by=order_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        ).all()
        objs = [self.row_as_obj(row) for row in result]
        return Ok(objs)

    def get_all_by_field(
        self,
        credentials: SyftVerifyKey,
        field_name: str,
        field_value: str,
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
    ) -> Result[list[SyftT], str]:
        result = self._get_by_fields(
            credentials=credentials,
            fields={field_name: field_value},
            order_by=order_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        ).all()
        objs = [self.row_as_obj(row) for row in result]
        return Ok(objs)

    def row_as_obj(self, row: Row) -> SyftT:
        return deserialize_json(row.fields, self.object_type)

    def get_role(self, credentials: SyftVerifyKey) -> ServiceRole:
        user_table = Table("User", Base.metadata)
        stmt = user_table.select().where(
            self._get_field_filter("verify_key", str(credentials), table=user_table),
        )
        result = self.session.execute(stmt).first()
        if result is None:
            return ServiceRole.GUEST
        return ServiceRole[result.fields["role"]]

    def _get_permission_filter(
        self,
        credentials: SyftVerifyKey,
        permission: ActionPermission = ActionPermission.READ,
    ) -> sa.sql.elements.BinaryExpression:
        role = self.get_role(credentials)
        if role in (ServiceRole.ADMIN, ServiceRole.DATA_OWNER):
            return sa.literal(True)

        permission_string = ActionObjectPermission(
            uid=UID(),  # dummy uid, we just need the permission string
            credentials=credentials,
            permission=permission,
        ).permission_string

        compound_permission_map = {
            ActionPermission.READ: ActionPermission.ALL_READ,
            ActionPermission.WRITE: ActionPermission.ALL_WRITE,
            ActionPermission.EXECUTE: ActionPermission.ALL_EXECUTE,
        }
        compound_permission_string = ActionObjectPermission(
            uid=UID(),  # dummy uid, we just need the permission string
            credentials=None,  # no credentials for compound permissions
            permission=compound_permission_map[permission],
        ).permission_string

        return sa.or_(
            self.table.c.permissions.contains(permission_string),
            self.table.c.permissions.contains(compound_permission_string),
        )

    def _apply_limit_offset(
        self,
        stmt: Select,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Any:
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt

    def _apply_order_by(
        self,
        stmt: Select,
        order_by: str | None = None,
        sort_order: str = "asc",
    ) -> Any:
        default_order_by = self.table.c.created_at
        default_order_by = (
            default_order_by.desc() if sort_order == "desc" else default_order_by
        )
        if order_by is None:
            return stmt.order_by(default_order_by)
        else:
            order_by_col = self.table.c.fields[order_by]
            order_by = order_by_col.desc() if sort_order == "desc" else order_by_col
            return stmt.order_by(order_by, default_order_by)

    def _apply_permission_filter(
        self,
        stmt: Select,
        credentials: SyftVerifyKey,
        has_permission: bool = False,
    ) -> Any:
        if not has_permission:
            stmt = stmt.where(self._get_permission_filter(credentials))
        return stmt

    def get_all(
        self,
        credentials: SyftVerifyKey,
        has_permission: bool = False,
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
    ) -> Result[list[SyftT], str]:
        stmt = self.table.select()

        stmt = self._apply_permission_filter(stmt, credentials, has_permission)
        stmt = self._apply_order_by(stmt, order_by, sort_order)
        stmt = self._apply_limit_offset(stmt, limit, offset)

        result = self.session.execute(stmt).all()
        objs = [self.row_as_obj(row) for row in result]
        return Ok(objs)

    def update(
        self,
        credentials: SyftVerifyKey,
        obj: SyftT,
        has_permission: bool = False,
    ) -> Result[SyftT, str]:
        """
        NOTE: We cannot do partial updates on the database,
        because we are using computed fields that are not known to the DB or ORM:
        - serialize_json will add computed fields to the JSON stored in the database
        - If we update a single field in the JSON, the computed fields can get out of sync.
        - To fix, we either need db-supported computed fields, or know in our ORM which fields should be re-computed.
        """

        # TODO has_permission is not used
        if not self.is_unique(obj):
            return Err(f"Some fields are not unique for {type(obj).__name__}")

        stmt = (
            self.table.update()
            .where(
                sa.and_(
                    self._get_field_filter("id", obj.id),
                    self._get_permission_filter(credentials, ActionPermission.WRITE),
                )
            )
            .values(fields=serialize_json(obj))
        )
        self.session.execute(stmt)
        self.session.commit()
        return Ok(obj)

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftT,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,  # TODO: check the default value
        ignore_duplicates: bool = False,  # only used in one place, should use upsert instead
    ) -> Result[SyftT, str]:
        # uid is unique by database constraint
        uid = obj.id

        if not self.is_unique(obj):
            return Err(f"Some fields are not unique for {type(obj).__name__}")

        permissions = self.get_ownership_permissions(uid, credentials)
        if add_permissions is not None:
            add_permission_strings = [p.permission_string for p in add_permissions]
            permissions.extend(add_permission_strings)

        storage_permissions = []
        if add_storage_permission:
            storage_permissions.append(
                StoragePermission(
                    uid=uid,
                    server_uid=self.server_uid,
                )
            )

            # TODO: write the storage permissions to the database

        # create the object with the permissions
        stmt = self.table.insert().values(
            id=uid,
            fields=serialize_json(obj),
            permissions=permissions,
            # storage_permissions=storage_permissions,
        )
        self.session.execute(stmt)
        self.session.commit()
        return Ok(obj)

    def get_ownership_permissions(
        self, uid: UID, credentials: SyftVerifyKey
    ) -> list[str]:
        return [
            ActionObjectOWNER(uid=uid, credentials=credentials).permission_string,
            ActionObjectWRITE(uid=uid, credentials=credentials).permission_string,
            ActionObjectREAD(uid=uid, credentials=credentials).permission_string,
            ActionObjectEXECUTE(uid=uid, credentials=credentials).permission_string,
        ]

    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> Result[SyftSuccess, str]:
        # TODO check delete permissions
        stmt = self.table.delete().where(
            sa.and_(
                self._get_field_filter("id", uid),
                self._get_permission_filter(credentials, ActionPermission.OWNER),
            )
        )
        self.session.execute(stmt)
        self.session.commit()
        return Ok(
            SyftSuccess(message=f"{type(self.object_type).__name__}: {uid} deleted")
        )

    def add_permissions(self, permissions: list[ActionObjectPermission]) -> None:
        # TODO: should do this in a single transaction
        for permission in permissions:
            self.add_permission(permission)
        return None

    def add_permission(self, permission: ActionObjectPermission) -> None:
        stmt = (
            self.table.update()
            .values(
                permissions=sa.func.array_append(
                    self.table.c.permissions, permission.permission_string
                )
            )
            .where(
                sa.and_(
                    self._get_field_filter("id", permission.uid),
                    self._get_permission_filter(
                        permission.credentials, ActionPermission.WRITE
                    ),
                )
            )
        )
        self.session.execute(stmt)
        self.session.commit()
        return None

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        stmt = (
            self.table.update()
            .values(
                permissions=sa.func.array_remove(self.table.c.permissions, permission)
            )
            .where(
                sa.and_(
                    self._get_field_filter("id", permission.uid),
                    self._get_permission_filter(
                        permission.credentials,
                        # since anyone with write permission can add permissions,
                        # owner check doesn't make sense, it should be write
                        ActionPermission.OWNER,
                    ),
                )
            )
        )
        self.session.execute(stmt)
        self.session.commit()
        return None

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        stmt = self.table.select().where(
            sa.and_(
                self._get_field_filter("id", permission.uid),
                self.table.c.permissions.contains(permission.permission_string),
            )
        )
        result = self.session.execute(stmt).first()
        return result is not None

    def has_storage_permission(self, permission: StoragePermission) -> bool:
        # TODO
        return True
