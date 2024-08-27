# stdlib

# stdlib
from typing import Any
from typing import Generic
from typing import get_args
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
from sqlalchemy import select
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
from ...service.user.user_roles import ServiceRole
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..document_store_errors import NotFoundException
from ..document_store_errors import StashException
from .models import Base
from .models import UIDTypeDecorator
from .sqlite_db import DBManager

SyftT = TypeVar("SyftT", bound=SyftObject)
T = TypeVar("T")


class ObjectStash(Generic[SyftT]):
    table: Table
    object_type: type

    def __init__(self, store: DBManager) -> None:
        self.db = store
        self.object_type = self.get_object_type()
        self.table = self._create_table()

    @classmethod
    def get_object_type(cls) -> type[SyftT]:
        generic_args = get_args(cls.__orig_bases__[0])
        if len(generic_args) != 1:
            raise TypeError("ObjectStash must have a single generic argument")
        elif not issubclass(generic_args[0], SyftObject):
            raise TypeError(
                "ObjectStash generic argument must be a subclass of SyftObject"
            )
        return generic_args[0]

    @property
    def server_uid(self) -> UID:
        return self.db.server_uid

    @property
    def root_verify_key(self) -> SyftVerifyKey:
        return self.db.root_verify_key

    @as_result(StashException)
    def check_type(self, obj: Any, type_: type) -> Any:
        if not isinstance(obj, type_):
            raise StashException(f"{type(obj)} does not match required type: {type_}")
        return obj

    @property
    def session(self) -> Session:
        return self.db.session

    def _create_table(self) -> Table:
        # need to call Base.metadata.create_all(engine) to create the table
        table_name = self.object_type.__canonical_name__
        if table_name not in Base.metadata.tables:
            Table(
                self.object_type.__canonical_name__,
                Base.metadata,
                Column("id", UIDTypeDecorator, primary_key=True, default=uuid.uuid4),
                Column("fields", JSON, default={}),
                Column("permissions", JSON, default=[]),
                Column("storage_permissions", JSON, default=[]),
                # TODO rename and use on SyftObject fields
                Column(
                    "created_at", sa.DateTime, server_default=sa.func.now(), index=True
                ),
                Column("updated_at", sa.DateTime, server_onupdate=sa.func.now()),
                Column("deleted_at", sa.DateTime, index=True),
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

    def exists(self, credentials: SyftVerifyKey, uid: UID) -> bool:
        # TODO needs credentials check?
        # TODO use COUNT(*) instead of SELECT
        stmt = self.table.select().where(self._get_field_filter("id", uid))
        result = self.session.execute(stmt).first()
        return result is not None

    @as_result(SyftException, StashException, NotFoundException)
    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> SyftT:
        stmt = self.table.select()
        stmt = stmt.where(self._get_field_filter("id", uid))
        stmt = self._apply_permission_filter(stmt, credentials, has_permission)
        result = self.session.execute(stmt).first()

        if result is None:
            raise NotFoundException(
                f"{type(self.object_type).__name__}: {uid} not found"
            )
        return self.row_as_obj(result)

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
        has_permission: bool = False,
    ) -> sa.Result:
        table = table if table is not None else self.table
        filters = []
        for field_name, field_value in fields.items():
            filt = self._get_field_filter(field_name, field_value, table=table)
            filters.append(filt)

        stmt = table.select()
        stmt = stmt.where(sa.and_(*filters))
        stmt = self._apply_permission_filter(
            stmt, credentials, has_permission=has_permission
        )
        stmt = self._apply_order_by(stmt, order_by, sort_order)
        stmt = self._apply_limit_offset(stmt, limit, offset)

        result = self.session.execute(stmt)
        return result

    @as_result(SyftException, StashException, NotFoundException)
    def get_one_by_field(
        self,
        credentials: SyftVerifyKey,
        field_name: str,
        field_value: str,
        has_permission: bool = False,
    ) -> SyftT:
        return self.get_one_by_fields(
            credentials=credentials,
            fields={field_name: field_value},
            has_permission=has_permission,
        )

    @as_result(SyftException, StashException, NotFoundException)
    def get_one_by_fields(
        self,
        credentials: SyftVerifyKey,
        fields: dict[str, str],
        has_permission: bool = False,
    ) -> SyftT:
        result = self._get_by_fields(
            credentials=credentials,
            fields=fields,
            has_permission=has_permission,
        ).first()
        if result is None:
            raise NotFoundException(f"{type(self.object_type).__name__}: not found")
        return self.row_as_obj(result)

    @as_result(SyftException, StashException, NotFoundException)
    def get_all_by_fields(
        self,
        credentials: SyftVerifyKey,
        fields: dict[str, str],
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
        has_permission: bool = False,
    ) -> list[SyftT]:
        result = self._get_by_fields(
            credentials=credentials,
            fields=fields,
            order_by=order_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
            has_permission=has_permission,
        ).all()

        return [self.row_as_obj(row) for row in result]

    @as_result(SyftException, StashException, NotFoundException)
    def get_all_by_field(
        self,
        credentials: SyftVerifyKey,
        field_name: str,
        field_value: str,
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
        has_permission: bool = False,
    ) -> list[SyftT]:
        return self.get_all_by_fields(
            credentials=credentials,
            fields={field_name: field_value},
            order_by=order_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
            has_permission=has_permission,
        )

    @as_result(SyftException, StashException, NotFoundException)
    def get_all_contains(
        self,
        credentials: SyftVerifyKey,
        field_name: str,
        field_value: str,
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
        has_permission: bool = False,
    ) -> list[SyftT]:
        # TODO write filter logic, merge with get_all

        stmt = self.table.select().where(
            self.table.c.fields[field_name].contains(func.json_quote(field_value)),
        )
        stmt = self._apply_permission_filter(stmt, credentials, has_permission)
        stmt = self._apply_order_by(stmt, order_by, sort_order)
        stmt = self._apply_limit_offset(stmt, limit, offset)

        result = self.session.execute(stmt).all()
        return [self.row_as_obj(row) for row in result]

    def row_as_obj(self, row: Row) -> SyftT:
        # TODO make unwrappable serde
        return deserialize_json(row.fields)

    def get_role(self, credentials: SyftVerifyKey) -> ServiceRole:
        # TODO error handling
        user_table = Table("User", Base.metadata)
        stmt = select(user_table.c.fields["role"]).where(
            self._get_field_filter("verify_key", str(credentials), table=user_table),
        )
        role = self.session.scalar(stmt)
        if role is None:
            return ServiceRole.GUEST
        return ServiceRole[role]

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
        permission: ActionPermission = ActionPermission.READ,
        has_permission: bool = False,
    ) -> Any:
        if not has_permission:
            stmt = stmt.where(
                self._get_permission_filter(credentials, permission=permission)
            )
        return stmt

    @as_result(StashException)
    def get_all(
        self,
        credentials: SyftVerifyKey,
        has_permission: bool = False,
        order_by: str | None = None,
        sort_order: str = "asc",
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[SyftT]:
        stmt = self.table.select()

        stmt = self._apply_permission_filter(stmt, credentials, has_permission)
        stmt = self._apply_order_by(stmt, order_by, sort_order)
        stmt = self._apply_limit_offset(stmt, limit, offset)

        result = self.session.execute(stmt).all()
        return [self.row_as_obj(row) for row in result]

    @as_result(StashException)
    def update(
        self,
        credentials: SyftVerifyKey,
        obj: SyftT,
        has_permission: bool = False,
    ) -> SyftT:
        """
        NOTE: We cannot do partial updates on the database,
        because we are using computed fields that are not known to the DB or ORM:
        - serialize_json will add computed fields to the JSON stored in the database
        - If we update a single field in the JSON, the computed fields can get out of sync.
        - To fix, we either need db-supported computed fields, or know in our ORM which fields should be re-computed.
        """

        # TODO has_permission is not used
        if not self.is_unique(obj):
            raise StashException(f"Some fields are not unique for {type(obj).__name__}")

        # TODO error handling
        has_permission_stmt = (
            self._get_permission_filter(credentials, ActionPermission.WRITE)
            if has_permission
            else sa.literal(True)
        )
        stmt = (
            self.table.update()
            .where(
                sa.and_(
                    self._get_field_filter("id", obj.id),
                    has_permission_stmt,
                )
            )
            .values(fields=serialize_json(obj))
        )
        self.session.execute(stmt)
        self.session.commit()

        return self.get_by_uid(credentials, obj.id).unwrap()

    def get_ownership_permissions(
        self, uid: UID, credentials: SyftVerifyKey
    ) -> list[str]:
        return [
            ActionObjectOWNER(uid=uid, credentials=credentials).permission_string,
            ActionObjectWRITE(uid=uid, credentials=credentials).permission_string,
            ActionObjectREAD(uid=uid, credentials=credentials).permission_string,
            ActionObjectEXECUTE(uid=uid, credentials=credentials).permission_string,
        ]

    @as_result(StashException)
    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> UID:
        stmt = self.table.delete().where(self._get_field_filter("id", uid))
        stmt = self._apply_permission_filter(
            stmt,
            credentials,
            has_permission,
            permission=ActionPermission.WRITE,
        )
        self.session.execute(stmt)
        self.session.commit()
        return uid

    def add_permissions(self, permissions: list[ActionObjectPermission]) -> None:
        # TODO: should do this in a single transaction
        for permission in permissions:
            self.add_permission(permission)
        return None

    def add_permission(self, permission: ActionObjectPermission) -> None:
        stmt = (
            self.table.update()
            .where(self.table.c.id == permission.uid)
            .values(
                permissions=func.json_insert(
                    self.table.c.permissions,
                    "$[#]",
                    permission.permission_string,
                )
            )
        )

        self.session.execute(stmt)
        self.session.commit()

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        try:
            permissions = self._get_permissions_for_uid(permission.uid)
            permissions.remove(permission.permission_string)
        except (NotFoundException, KeyError):
            # TODO add error handling to permissions
            return None

        stmt = (
            self.table.update()
            .where(self.table.c.id == permission.uid)
            .values(permissions=list(permissions))
        )
        self.session.execute(stmt)
        self.session.commit()
        return None

    def remove_storage_permission(self, permission: StoragePermission) -> None:
        try:
            permissions = self._get_storage_permissions_for_uid(permission.uid)
            permissions.remove(permission.server_uid)
        except (NotFoundException, KeyError):
            # TODO add error handling to permissions
            return None

        stmt = (
            self.table.update()
            .where(self.table.c.id == permission.uid)
            .values(storage_permissions=[str(uid) for uid in permissions])
        )
        self.session.execute(stmt)
        self.session.commit()
        return None

    def _get_storage_permissions_for_uid(self, uid: UID) -> set[UID]:
        stmt = select(self.table.c.id, self.table.c.storage_permissions).where(
            self.table.c.id == uid
        )
        result = self.session.execute(stmt).first()
        if result is None:
            raise NotFoundException(f"No storage permissions found for uid: {uid}")
        return {UID(uid) for uid in result.storage_permissions}

    def get_all_storage_permissions(self) -> Result[dict[UID, set[UID]], str]:
        stmt = select(self.table.c.id, self.table.c.storage_permissions)
        results = self.session.execute(stmt).all()

        # make uid
        return {
            row.id: {(UID(uid) for uid in row.storage_permissions)} for row in results
        }

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        return self.has_permissions([permission])

    def has_storage_permission(self, permission: StoragePermission) -> bool:
        return self.has_storage_permissions([permission])

    def has_storage_permissions(self, permissions: list[StoragePermission]) -> bool:
        permission_filters = [
            sa.and_(
                self._get_field_filter("id", p.uid),
                self.table.c.storage_permissions.contains(p.server_uid),
            )
            for p in permissions
        ]

        stmt = self.table.select().where(
            sa.and_(
                *permission_filters,
            )
        )
        result = self.session.execute(stmt).first()
        return result

    def has_permissions(self, permissions: list[ActionObjectPermission]) -> bool:
        # NOTE: maybe we should use a permissions table to check all permissions at once
        # TODO: should check for compound permissions
        permission_filters = [
            sa.and_(
                self._get_field_filter("id", p.uid),
                self.table.c.permissions.contains(p.permission_string),
            )
            for p in permissions
        ]

        stmt = self.table.select().where(
            sa.and_(
                *permission_filters,
            )
        )
        result = self.session.execute(stmt).first()
        return result is not None

    def add_storage_permission(self, permission: StoragePermission) -> None:
        stmt = (
            self.table.update()
            .where(self.table.c.id == permission.uid)
            .values(
                storage_permissions=func.json_insert(
                    self.table.c.storage_permissions,
                    "$[#]",
                    permission.permission_string,
                )
            )
        )
        self.session.execute(stmt)
        self.session.commit()
        return None

    def add_storage_permissions(self, permissions: list[StoragePermission]) -> None:
        for permission in permissions:
            self.add_storage_permission(permission)

    def _get_permissions_for_uid(self, uid: UID) -> set[str]:
        stmt = select(self.table.c.permissions).where(self.table.c.id == uid)
        result = self.session.execute(stmt).scalar_one_or_none()
        if result is None:
            return NotFoundException(f"No permissions found for uid: {uid}")
        return set(result)

    def get_all_permissions(self) -> Result[dict[UID, set[str]], str]:
        stmt = select(self.table.c.id, self.table.c.permissions)
        results = self.session.execute(stmt).all()
        return Ok({row.id: set(row.permissions) for row in results})

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftT,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,  # TODO: check the default value
        ignore_duplicates: bool = False,
    ) -> Result[SyftT, str]:
        # uid is unique by database constraint
        uid = obj.id

        if self.exists(credentials, uid) or not self.is_unique(obj):
            if ignore_duplicates:
                return Ok(obj)
            return Err(f"Some fields are not unique for {type(obj).__name__}")

        permissions = self.get_ownership_permissions(uid, credentials)
        if add_permissions is not None:
            add_permission_strings = [p.permission_string for p in add_permissions]
            permissions.extend(add_permission_strings)

        storage_permissions = []
        if add_storage_permission:
            storage_permissions.append(
                self.server_uid,
            )

        # create the object with the permissions
        stmt = self.table.insert().values(
            id=uid,
            fields=serialize_json(obj),
            permissions=permissions,
            storage_permissions=storage_permissions,
        )
        self.session.execute(stmt)
        self.session.commit()
        return Ok(obj)
