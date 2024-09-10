# stdlib
from typing import Any
from typing import Generic
from typing import cast
from typing import get_args

# third party
import sqlalchemy as sa
from sqlalchemy import Row
from sqlalchemy import Table
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing_extensions import Self
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
from .query import PostgresQuery
from .query import Query
from .query import SQLiteQuery
from .schema import Base
from .schema import create_table
from .sqlite_db import DBManager
from .sqlite_db import SQLiteDBManager

StashT = TypeVar("StashT", bound=SyftObject)
T = TypeVar("T")


def parse_filters(filter_dict: dict[str, Any] | None) -> list[tuple[str, str, Any]]:
    # NOTE using django style filters, e.g. {"age__gt": 18}
    if filter_dict is None:
        return []
    filters = []
    for key, value in filter_dict.items():
        key_split = key.split("__")
        # Operator is eq if not specified
        if len(key_split) == 1:
            field, operator = key, "eq"
        elif len(key_split) == 2:
            field, operator = key_split
        filters.append((field, operator, value))
    return filters


class ObjectStash(Generic[StashT]):
    table: Table
    object_type: type[SyftObject]
    allow_any_type: bool = False

    def __init__(self, store: DBManager) -> None:
        self.db = store
        self.object_type = self.get_object_type()
        self.table = create_table(self.object_type, self.dialect)

    @property
    def dialect(self) -> sa.engine.interfaces.Dialect:
        return self.db.engine.dialect

    @classmethod
    def get_object_type(cls) -> type[StashT]:
        """
        Get the object type this stash is storing. This is the generic argument of the
        ObjectStash class.
        """
        generic_args = get_args(cls.__orig_bases__[0])
        if len(generic_args) != 1:
            raise TypeError("ObjectStash must have a single generic argument")
        elif not issubclass(generic_args[0], SyftObject):
            raise TypeError(
                "ObjectStash generic argument must be a subclass of SyftObject"
            )
        return generic_args[0]

    def __len__(self) -> int:
        return self.session.query(self.table).count()

    @classmethod
    def random(cls, **kwargs: dict) -> Self:
        db_manager = SQLiteDBManager.random(**kwargs)
        stash = cls(store=db_manager)
        stash.db.init_tables()
        return stash

    @property
    def server_uid(self) -> UID:
        return self.db.server_uid

    @property
    def root_verify_key(self) -> SyftVerifyKey:
        return self.db.root_verify_key

    @property
    def _data(self) -> list[StashT]:
        return self.get_all(self.root_verify_key, has_permission=True).unwrap()

    def query(self) -> Query:
        """Creates a query for this stash's object type."""
        if self.dialect.name == "sqlite":
            return SQLiteQuery(self.object_type)
        elif self.dialect.name == "postgresql":
            return PostgresQuery(self.object_type)
        else:
            raise NotImplementedError(f"Query not implemented for {self.dialect.name}")

    @as_result(StashException)
    def check_type(self, obj: T, type_: type) -> T:
        if not isinstance(obj, type_):
            raise StashException(f"{type(obj)} does not match required type: {type_}")
        return cast(T, obj)

    @property
    def session(self) -> Session:
        return self.db.session

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
        query = self.query().filter("id", "eq", uid)

        if not has_permission:
            role = self.get_role(credentials)
            query = query.with_permissions(credentials, role)

        result = query.execute(self.session).first()
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
            return table.c.fields[field_name].astext == json_value

    @as_result(SyftException, StashException, NotFoundException)
    def get_index(
        self, credentials: SyftVerifyKey, index: int, has_permission: bool = False
    ) -> StashT:
        order_by, sort_order = self.object_type.__order_by__
        if index < 0:
            index = -1 - index
            sort_order = "desc" if sort_order == "asc" else "asc"

        items = self.get_all(
            credentials,
            has_permission=has_permission,
            limit=1,
            offset=index,
            order_by=order_by,
            sort_order=sort_order,
        ).unwrap()

        if len(items) == 0:
            raise NotFoundException(f"No item found at index {index}")
        return items[0]

    def row_as_obj(self, row: Row) -> StashT:
        # TODO make unwrappable serde
        return deserialize_json(row.fields)

    def get_role(self, credentials: SyftVerifyKey) -> ServiceRole:
        # TODO error handling
        if Base.metadata.tables.get("User") is None:
            # if User table does not exist, we assume the user is a guest
            # this happens when we create stashes in tests
            return ServiceRole.GUEST
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

    @as_result(StashException, NotFoundException)
    def update(
        self,
        credentials: SyftVerifyKey,
        obj: StashT,
        has_permission: bool = False,
    ) -> StashT:
        """
        NOTE: We cannot do partial updates on the database,
        because we are using computed fields that are not known to the DB:
        - serialize_json will add computed fields to the JSON stored in the database
        - If we update a single field in the JSON, the computed fields can get out of sync.
        - To fix, we either need db-supported computed fields, or know in our ORM which fields should be re-computed.
        """

        if not self.allow_any_type:
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
        stmt = self.table.update().where(self.table.c.id == permission.uid)
        if self._is_sqlite():
            stmt = stmt.values(
                permissions=func.json_insert(
                    self.table.c.permissions,
                    "$[#]",
                    permission.permission_string,
                )
            )
        else:
            stmt = stmt.values(
                permissions=func.array_append(
                    self.table.c.permissions, permission.permission_string
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
            UID(row.id): {UID(uid) for uid in row.storage_permissions}
            for row in results
        }

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        if self.get_role(permission.credentials) in (
            ServiceRole.ADMIN,
            ServiceRole.DATA_OWNER,
        ):
            return True
        return self.has_permissions([permission])

    def has_storage_permission(self, permission: StoragePermission) -> bool:
        return self.has_storage_permissions([permission])

    def _is_sqlite(self) -> bool:
        return self.db.engine.dialect.name == "sqlite"

    def has_storage_permissions(self, permissions: list[StoragePermission]) -> bool:
        permission_filters = [
            sa.and_(
                self._get_field_filter("id", p.uid),
                self.table.c.storage_permissions.contains(
                    p.server_uid.no_dash
                    if self._is_sqlite()
                    else [p.server_uid.no_dash]
                ),
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
        stmt = self.table.update().where(self.table.c.id == permission.uid)
        if self._is_sqlite():
            stmt = stmt.values(
                storage_permissions=func.json_insert(
                    self.table.c.storage_permissions,
                    "$[#]",
                    permission.permission_string,
                )
            )
        else:
            stmt = stmt.values(
                permissions=func.array_append(
                    self.table.c.storage_permissions, permission.server_uid.no_dash
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
                self.server_uid.no_dash,
            )

        fields = serialize_json(obj)
        try:
            # check if the fields are deserializable
            # PR NOTE: Is this too much extra work?
            deserialize_json(fields)
        except Exception as e:
            raise StashException(
                f"Error serializing object: {e}. Some fields are invalid."
            )

        # create the object with the permissions
        stmt = self.table.insert().values(
            id=uid,
            fields=fields,
            permissions=permissions,
            storage_permissions=storage_permissions,
        )
        self.session.execute(stmt)
        self.session.commit()
        return self.get_by_uid(credentials, uid).unwrap()

    @as_result(StashException)
    def get_one(
        self,
        credentials: SyftVerifyKey,
        filters: dict[str, Any] | None = None,
        has_permission: bool = False,
        order_by: str | None = None,
        sort_order: str | None = None,
        offset: int = 0,
    ) -> StashT:
        result = self.get_all(
            credentials=credentials,
            filters=filters,
            has_permission=has_permission,
            order_by=order_by,
            sort_order=sort_order,
            limit=1,
            offset=offset,
        ).unwrap()
        if len(result) == 0:
            raise NotFoundException(f"{self.object_type.__name__}: not found")
        return result[0]

    @as_result(StashException)
    def get_all(
        self,
        credentials: SyftVerifyKey,
        filters: dict[str, Any] | None = None,
        has_permission: bool = False,
        order_by: str | None = None,
        sort_order: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StashT]:
        """
        Get all objects from the stash, optionally filtered.

        Args:
            credentials (SyftVerifyKey): credentials of the user
            filters (dict[str, Any] | None, optional): dictionary of filters,
                where the key is the field name and the value is the filter value.
                Operators other than equals can be used in the key,
                e.g. {"name": "Bob", "friends__contains": "Alice"}. Defaults to None.
            has_permission (bool, optional): If True, overrides the permission check.
                Defaults to False.
            order_by (str | None, optional): If provided, the results will be ordered by this field.
                If not provided, the default order and field defined on the SyftObject.__order_by__ are used.
                Defaults to None.
            sort_order (str | None, optional): "asc" or "desc" If not defined,
                the default order defined on the SyftObject.__order_by__ is used.
                Defaults to None.
            limit (int | None, optional): limit the number of results. Defaults to None.
            offset (int, optional): offset the results. Defaults to 0.

        Returns:
            list[StashT]: list of objects.
        """
        query = self.query()

        if not has_permission:
            role = self.get_role(credentials)
            query = query.with_permissions(credentials, role)

        for field_name, operator, field_value in parse_filters(filters):
            query = query.filter(field_name, operator, field_value)

        query = query.order_by(order_by, sort_order).limit(limit).offset(offset)
        result = query.execute(self.session).all()

        return [self.row_as_obj(row) for row in result]
