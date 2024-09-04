# stdlib

# stdlib
from functools import cache
from typing import Any
from typing import Generic
from typing import cast
from typing import get_args
import uuid

# third party
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Row
from sqlalchemy import Table
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
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

StashT = TypeVar("StashT", bound=SyftObject)
T = TypeVar("T")


class ObjectStash(Generic[StashT]):
    table: Table
    object_type: type[SyftObject]
    allow_any_type: bool = False

    def __init__(self, store: DBManager) -> None:
        self.db = store
        self.object_type = self.get_object_type()
        self.table = self._create_table()

    @classmethod
    def get_object_type(cls) -> type[StashT]:
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

    @property
    def _data(self) -> list[StashT]:
        return self.get_all(self.root_verify_key, has_permission=True).unwrap()

    @as_result(StashException)
    def check_type(self, obj: T, type_: type) -> T:
        if not isinstance(obj, type_):
            raise StashException(f"{type(obj)} does not match required type: {type_}")
        return cast(T, obj)

    @property
    def session(self) -> Session:
        return self.db.session

    def _create_table(self) -> Table:
        # need to call Base.metadata.create_all(engine) to create the table
        table_name = self.object_type.__canonical_name__

        fields_type = (
            JSON if self.db.engine.dialect.name == "sqlite" else postgresql.JSONB
        )
        permissons_type = (
            JSON
            if self.db.engine.dialect.name == "sqlite"
            else postgresql.ARRAY(sa.String)
        )
        storage_permissions_type = (
            JSON
            if self.db.engine.dialect.name == "sqlite"
            else postgresql.ARRAY(UIDTypeDecorator)
        )
        if table_name not in Base.metadata.tables:
            Table(
                self.object_type.__canonical_name__,
                Base.metadata,
                Column("id", UIDTypeDecorator, primary_key=True, default=uuid.uuid4),
                Column("fields", fields_type, default={}),
                Column("permissions", permissons_type, default=[]),
                Column(
                    "storage_permissions",
                    storage_permissions_type,
                    default=[],
                ),
                # TODO rename and use on SyftObject fields
                Column(
                    "_created_at", sa.DateTime, server_default=sa.func.now(), index=True
                ),
                Column("_updated_at", sa.DateTime, server_onupdate=sa.func.now()),
                Column("_deleted_at", sa.DateTime, index=True),
            )
        return Base.metadata.tables[table_name]

    def _drop_table(self) -> None:
        table_name = self.object_type.__canonical_name__
        if table_name in Base.metadata.tables:
            Base.metadata.tables[table_name].drop(self.db.engine)
        else:
            raise StashException(f"Table {table_name} does not exist")

    def _print_query(self, stmt: sa.sql.select) -> None:
        print(
            stmt.compile(
                compile_kwargs={"literal_binds": True},
                dialect=self.session.bind.dialect,
            )
        )

    @property
    def unique_fields(self) -> list[str]:
        return getattr(self.object_type, "__attr_unique__", [])

    def is_unique(self, obj: StashT) -> bool:
        unique_fields = self.unique_fields
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
    ) -> StashT:
        stmt = self.table.select()
        stmt = stmt.where(self._get_field_filter("id", uid))
        stmt = self._apply_permission_filter(
            stmt, credentials=credentials, has_permission=has_permission
        )

        result = self.session.execute(stmt).first()

        if result is None:
            raise NotFoundException(f"{self.object_type.__name__}: {uid} not found")
        return self.row_as_obj(result)

    def _get_field_filter(
        self,
        field_name: str,
        field_value: Any,
        table: Table | None = None,
    ) -> sa.sql.elements.BinaryExpression:
        table = table if table is not None else self.table
        if field_name == "id":
            uid_field_value = UID(field_value)
            return table.c.id == uid_field_value

        json_value = serialize_json(field_value)
        if self.db.engine.dialect.name == "sqlite":
            return table.c.fields[field_name] == func.json_quote(json_value)
        elif self.db.engine.dialect.name == "postgresql":
            return sa.cast(table.c.fields[field_name], sa.String) == json_value

    def _get_by_fields(
        self,
        credentials: SyftVerifyKey,
        fields: dict[str, str],
        table: Table | None = None,
        order_by: str | None = None,
        sort_order: str | None = None,
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
            stmt, credentials=credentials, has_permission=has_permission
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
    ) -> StashT:
        return self.get_one_by_fields(
            credentials=credentials,
            fields={field_name: field_value},
            has_permission=has_permission,
        ).unwrap()

    @as_result(SyftException, StashException, NotFoundException)
    def get_one_by_fields(
        self,
        credentials: SyftVerifyKey,
        fields: dict[str, str],
        has_permission: bool = False,
    ) -> StashT:
        result = self._get_by_fields(
            credentials=credentials,
            fields=fields,
            has_permission=has_permission,
        ).first()
        if result is None:
            raise NotFoundException(f"{self.object_type.__name__}: not found")
        return self.row_as_obj(result)

    @as_result(SyftException, StashException, NotFoundException)
    def get_all_by_fields(
        self,
        credentials: SyftVerifyKey,
        fields: dict[str, str],
        order_by: str | None = None,
        sort_order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        has_permission: bool = False,
    ) -> list[StashT]:
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
        sort_order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        has_permission: bool = False,
    ) -> list[StashT]:
        return self.get_all_by_fields(
            credentials=credentials,
            fields={field_name: field_value},
            order_by=order_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
            has_permission=has_permission,
        ).unwrap()

    @as_result(SyftException, StashException, NotFoundException)
    def get_all_contains(
        self,
        credentials: SyftVerifyKey,
        field_name: str,
        field_value: str,
        order_by: str | None = None,
        sort_order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        has_permission: bool = False,
    ) -> list[StashT]:
        # TODO write filter logic, merge with get_all

        stmt = self.table.select().where(
            self.table.c.fields[field_name].contains(func.json_quote(field_value)),
        )
        stmt = self._apply_permission_filter(
            stmt, credentials=credentials, has_permission=has_permission
        )
        stmt = self._apply_order_by(stmt, order_by, sort_order)
        stmt = self._apply_limit_offset(stmt, limit, offset)

        result = self.session.execute(stmt).all()
        return [self.row_as_obj(row) for row in result]

    @as_result(SyftException, StashException, NotFoundException)
    def get_index(
        self, credentials: SyftVerifyKey, index: int, has_permission: bool = False
    ) -> StashT:
        items = self.get_all(
            credentials,
            has_permission=has_permission,
            limit=1,
            offset=index,
        ).unwrap()

        if len(items) == 0:
            raise NotFoundException(f"No item found at index {index}")
        return items[0]

    def row_as_obj(self, row: Row) -> StashT:
        # TODO make unwrappable serde
        return deserialize_json(row.fields)

    # TODO add cache invalidation, ignore B019
    @cache  # noqa: B019
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

    def _get_permission_filter_from_permisson(
        self,
        permission: ActionObjectPermission,
    ) -> sa.sql.elements.BinaryExpression:
        permission_string = permission.permission_string
        compound_permission_string = permission.compound_permission_string

        if self.session.bind.dialect.name == "postgresql":
            permission_string = [permission_string]  # type: ignore
            compound_permission_string = [compound_permission_string]  # type: ignore
        return sa.or_(
            self.table.c.permissions.contains(permission_string),
            self.table.c.permissions.contains(compound_permission_string),
        )

    def _apply_limit_offset(
        self,
        stmt: T,
        limit: int | None = None,
        offset: int | None = None,
    ) -> T:
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt

    def _get_order_by_col(self, order_by: str, sort_order: str | None = None) -> Column:
        # TODO connect+rename created_date to created_at
        if sort_order is None:
            sort_order = "asc"

        if order_by == "id":
            col = self.table.c.id
        if order_by == "created_date" or order_by == "_created_at":
            col = self.table.c._created_at
        else:
            col = self.table.c.fields[order_by]

        return col.desc() if sort_order.lower() == "desc" else col.asc()

    def _apply_order_by(
        self,
        stmt: T,
        order_by: str | None = None,
        sort_order: str | None = None,
    ) -> T:
        if order_by is None:
            order_by, default_sort_order = self.object_type.__order_by__
            sort_order = sort_order or default_sort_order

        order_by_col = self._get_order_by_col(order_by, sort_order)

        if order_by == "id":
            return stmt.order_by(order_by_col)
        else:
            secondary_order_by = self._get_order_by_col("id", sort_order)
            return stmt.order_by(order_by_col, secondary_order_by)

    def _apply_permission_filter(
        self,
        stmt: T,
        *,
        credentials: SyftVerifyKey,
        permission: ActionPermission = ActionPermission.READ,
        has_permission: bool = False,
    ) -> T:
        if has_permission:
            # ignoring permissions
            return stmt
        role = self.get_role(credentials)
        if role in (ServiceRole.ADMIN, ServiceRole.DATA_OWNER):
            # admins and data owners have all permissions
            return stmt

        action_object_permission = ActionObjectPermission(
            uid=UID(),  # dummy uid, we just need the permission string
            credentials=credentials,
            permission=permission,
        )

        stmt = stmt.where(
            self._get_permission_filter_from_permisson(
                permission=action_object_permission
            )
        )
        return stmt

    @as_result(StashException)
    def get_all(
        self,
        credentials: SyftVerifyKey,
        has_permission: bool = False,
        order_by: str | None = None,
        sort_order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[StashT]:
        stmt = self.table.select()

        stmt = self._apply_permission_filter(
            stmt,
            credentials=credentials,
            has_permission=has_permission,
            permission=ActionPermission.READ,
        )
        stmt = self._apply_order_by(stmt, order_by, sort_order)
        stmt = self._apply_limit_offset(stmt, limit, offset)

        result = self.session.execute(stmt).all()
        return [self.row_as_obj(row) for row in result]

    @as_result(StashException, NotFoundException)
    def update(
        self,
        credentials: SyftVerifyKey,
        obj: StashT,
        has_permission: bool = False,
    ) -> StashT:
        """
        NOTE: We cannot do partial updates on the database,
        because we are using computed fields that are not known to the DB or ORM:
        - serialize_json will add computed fields to the JSON stored in the database
        - If we update a single field in the JSON, the computed fields can get out of sync.
        - To fix, we either need db-supported computed fields, or know in our ORM which fields should be re-computed.
        """

        self.check_type(obj, self.object_type).unwrap()

        # TODO has_permission is not used
        if not self.is_unique(obj):
            raise StashException(f"Some fields are not unique for {type(obj).__name__}")

        stmt = self.table.update().where(self._get_field_filter("id", obj.id))
        stmt = self._apply_permission_filter(
            stmt,
            credentials=credentials,
            permission=ActionPermission.WRITE,
            has_permission=has_permission,
        )
        fields = serialize_json(obj)
        try:
            deserialize_json(fields)
        except Exception as e:
            raise StashException(
                f"Error serializing object: {e}. Some fields are invalid."
            )
        stmt = stmt.values(fields=fields)

        result = self.session.execute(stmt)
        self.session.commit()
        if result.rowcount == 0:
            raise NotFoundException(
                f"{self.object_type.__name__}: {obj.id} not found or no permission to update."
            )
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

    @as_result(StashException, NotFoundException)
    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> UID:
        stmt = self.table.delete().where(self._get_field_filter("id", uid))
        stmt = self._apply_permission_filter(
            stmt,
            credentials=credentials,
            permission=ActionPermission.WRITE,
            has_permission=has_permission,
        )
        result = self.session.execute(stmt)
        self.session.commit()
        if result.rowcount == 0:
            raise NotFoundException(
                f"{self.object_type.__name__}: {uid} not found or no permission to delete."
            )
        return uid

    @as_result(NotFoundException)
    def add_permissions(self, permissions: list[ActionObjectPermission]) -> None:
        # TODO: should do this in a single transaction
        # TODO add error handling
        for permission in permissions:
            self.add_permission(permission).unwrap()
        return None

    @as_result(NotFoundException)
    def add_permission(self, permission: ActionObjectPermission) -> None:
        # TODO add error handling
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

        result = self.session.execute(stmt)
        self.session.commit()
        if result.rowcount == 0:
            raise NotFoundException(
                f"{self.object_type.__name__}: {permission.uid} not found or no permission to change."
            )

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        # TODO not threadsafe
        try:
            permissions = self._get_permissions_for_uid(permission.uid).unwrap()
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
        # TODO not threadsafe
        try:
            permissions = self._get_storage_permissions_for_uid(permission.uid).unwrap()
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

    @as_result(StashException)
    def _get_storage_permissions_for_uid(self, uid: UID) -> set[UID]:
        stmt = select(self.table.c.id, self.table.c.storage_permissions).where(
            self.table.c.id == uid
        )
        result = self.session.execute(stmt).first()
        if result is None:
            raise NotFoundException(f"No storage permissions found for uid: {uid}")
        return {UID(uid) for uid in result.storage_permissions}

    @as_result(StashException)
    def get_all_storage_permissions(self) -> dict[UID, set[UID]]:
        stmt = select(self.table.c.id, self.table.c.storage_permissions)
        results = self.session.execute(stmt).all()

        return {
            UID(row.id): {(UID(uid) for uid in row.storage_permissions)}
            for row in results
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
        return result is not None

    def has_permissions(self, permissions: list[ActionObjectPermission]) -> bool:
        # TODO: we should use a permissions table to check all permissions at once
        # TODO: should check for compound permissions

        permission_filters = [
            sa.and_(
                self._get_field_filter("id", p.uid),
                self._get_permission_filter_from_permisson(permission=p),
            )
            for p in permissions
        ]

        stmt = self.table.select().where(
            sa.and_(
                *permission_filters,
            ),
        )
        result = self.session.execute(stmt).first()
        return result is not None

    @as_result(NotFoundException)
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
        result = self.session.execute(stmt)
        self.session.commit()
        if result.rowcount == 0:
            raise NotFoundException(
                f"{self.object_type.__name__}: {permission.uid} not found or no permission to change."
            )
        return None

    @as_result(NotFoundException)
    def add_storage_permissions(self, permissions: list[StoragePermission]) -> None:
        for permission in permissions:
            self.add_storage_permission(permission)

    @as_result(StashException)
    def _get_permissions_for_uid(self, uid: UID) -> set[str]:
        stmt = select(self.table.c.permissions).where(self.table.c.id == uid)
        result = self.session.execute(stmt).scalar_one_or_none()
        if result is None:
            return NotFoundException(f"No permissions found for uid: {uid}")
        return set(result)

    @as_result(StashException)
    def get_all_permissions(self) -> dict[UID, set[str]]:
        stmt = select(self.table.c.id, self.table.c.permissions)
        results = self.session.execute(stmt).all()
        return {UID(row.id): set(row.permissions) for row in results}

    @as_result(SyftException, StashException)
    def set(
        self,
        credentials: SyftVerifyKey,
        obj: StashT,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,  # TODO: check the default value
        ignore_duplicates: bool = False,
    ) -> StashT:
        if not self.allow_any_type:
            self.check_type(obj, self.object_type).unwrap()
        uid = obj.id

        # check if the object already exists
        if self.exists(credentials, uid) or not self.is_unique(obj):
            if ignore_duplicates:
                return obj
            unique_fields_str = ", ".join(self.unique_fields)
            raise StashException(
                public_message=f"Duplication Key Error for {obj}.\n"
                f"The fields that should be unique are {unique_fields_str}."
            )

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
        return self.get_by_uid(credentials, uid).unwrap()
