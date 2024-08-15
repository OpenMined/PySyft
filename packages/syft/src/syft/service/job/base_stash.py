# stdlib

# stdlib
import base64
from enum import Enum
import json
import threading
from typing import Any
from typing import Generic
import uuid
from uuid import UUID

# third party
import pydantic
from result import Err
from result import Ok
from result import Result
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Row
from sqlalchemy import Table
from sqlalchemy import TypeDecorator
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import Session
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import JSON
from typing_extensions import TypeVar

# syft absolute
import syft as sy

# relative
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..action.action_permissions import ActionObjectEXECUTE
from ..action.action_permissions import ActionObjectOWNER
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionObjectREAD
from ..action.action_permissions import ActionObjectWRITE
from ..action.action_permissions import ActionPermission
from ..action.action_permissions import StoragePermission
from ..response import SyftSuccess
from ..user.user_roles import ServiceRole


class Base(DeclarativeBase):
    pass


class UIDTypeDecorator(TypeDecorator):
    """Converts between Syft UID and UUID."""

    impl = sa.UUID
    cache_ok = True

    def process_bind_param(self, value, dialect):  # type: ignore
        if value is not None:
            return value

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


def should_handle_as_bytes(type_) -> bool:
    # relative
    from ...util.misc_objs import HTMLObject
    from ...util.misc_objs import MarkdownDescription
    from ..action.action_object import Action
    from ..dataset.dataset import Asset
    from ..dataset.dataset import Contributor
    from ..request.request import Change
    from ..request.request import ChangeStatus
    from ..settings.settings import PwdTokenResetConfig

    return (
        type_.annotation is LinkedObject
        or type_.annotation == LinkedObject | None
        or type_.annotation == list[UID] | dict[str, UID] | None
        or type_.annotation == dict[str, UID] | None
        or type_.annotation == list[Change]
        or type_.annotation == Any | None  # type: ignore
        or type_.annotation == Action | None  # type: ignore
        or getattr(type_.annotation, "__origin__", None) is dict
        or type_.annotation == HTMLObject | MarkdownDescription
        or type_.annotation == PwdTokenResetConfig
        or type_.annotation == list[ChangeStatus]
        or type_.annotation == list[Asset]
        or type_.annotation == set[Contributor]
        or type_.annotation == MarkdownDescription
        or type_.annotation == Contributor
    )


def model_dump(obj: pydantic.BaseModel) -> dict:
    obj_dict = dict(obj)  # obj.model_dump() does not work when
    for key, type_ in obj.model_fields.items():
        if type_.annotation is UID:
            obj_dict[key] = obj_dict[key].no_dash
        elif (
            type_.annotation is SyftVerifyKey
            or type_.annotation == SyftVerifyKey | None
            or type_.annotation is SyftSigningKey
            or type_.annotation == SyftSigningKey | None
        ):
            attr = getattr(obj, key)
            obj_dict[key] = str(attr) if attr is not None else None
        elif type_.annotation is DateTime or type_.annotation == DateTime | None:
            # FIXME: this is a hack, we should not be converting to string
            if obj_dict[key] is not None:
                obj_dict[key] = obj_dict[key].utc_timestamp
        elif should_handle_as_bytes(type_):
            # not very efficient as it serializes the object twice
            data = sy.serialize(getattr(obj, key), to_bytes=True)
            base64_data = base64.b64encode(data).decode("utf-8")
            obj_dict[key] = base64_data

    return obj_dict


T = TypeVar("T", bound=pydantic.BaseModel)


def model_validate(obj_type: type[T], obj_dict: dict) -> T:
    # relative

    for key, type_ in obj_type.model_fields.items():
        if key not in obj_dict:
            continue
        # FIXME
        if type_.annotation is UID or type_.annotation == UID | None:
            if obj_dict[key] is None:
                obj_dict[key] = None
            else:
                obj_dict[key] = UID(obj_dict[key])
        elif (
            type_.annotation is SyftVerifyKey
            or type_.annotation == SyftVerifyKey | None
        ):
            if obj_dict[key] is None:
                obj_dict[key] = None
            elif isinstance(obj_dict[key], str):
                obj_dict[key] = SyftVerifyKey.from_string(obj_dict[key])
            elif isinstance(obj_dict[key], SyftVerifyKey):
                obj_dict[key] = obj_dict[key]
        elif type_.annotation is DateTime or type_.annotation == DateTime | None:
            if obj_dict[key] is not None:
                obj_dict[key] = DateTime.from_timestamp(obj_dict[key])
        elif (
            type_.annotation is SyftSigningKey
            or type_.annotation == SyftSigningKey | None
        ):
            if obj_dict[key] is None:
                obj_dict[key] = None
            elif isinstance(obj_dict[key], str):
                obj_dict[key] = SyftSigningKey(signing_key=obj_dict[key])
            elif isinstance(obj_dict[key], SyftSigningKey):
                obj_dict[key] = obj_dict[key]
        elif should_handle_as_bytes(type_):
            data = base64.b64decode(obj_dict[key])
            obj_dict[key] = sy.deserialize(data, from_bytes=True)

    return obj_type.model_validate(obj_dict)


def _default_dumps(val):  # type: ignore
    if isinstance(val, UID):
        return str(val.no_dash)
    elif isinstance(val, UUID):
        return val.hex
    elif issubclass(type(val), Enum):
        return val.name
    elif val is None:
        return None
    return str(val)
    # elif isinstance
    raise TypeError(f"Can't serialize {val}, type {type(val)}")


def _default_loads(val):  # type: ignore
    if "UID" in val:
        return UID(val)
    return val


def dumps(d: dict) -> str:
    return json.dumps(d, default=_default_dumps)


def loads(d: str) -> dict:
    return json.loads(d, object_hook=_default_loads)


class SQLiteDBManager:
    def __init__(self, server_uid: str) -> None:
        self.server_uid = server_uid
        self.path = f"sqlite:////tmp/{server_uid}.db"
        self.engine = create_engine(
            self.path, json_serializer=dumps, json_deserializer=loads
        )
        print(f"Connecting to {self.path}")
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.thread_local = threading.local()

        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        if not hasattr(self.thread_local, "session"):
            self.thread_local.session = self.SessionFactory()
        return self.thread_local.session

    @property
    def session(self) -> Session:
        return self.get_session()


SyftT = TypeVar("SyftT", bound=SyftObject)


class ObjectStash(Generic[SyftT]):
    object_type: type[SyftT]

    def __init__(self, store: DocumentStore) -> None:
        self.server_uid = store.server_uid
        self.root_verify_key = store.root_verify_key
        # is there a better way to init the table
        _ = self.table
        self.db = SQLiteDBManager(self.server_uid)

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
                Column("created_at", sa.DateTime, server_default=sa.func.now()),
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

    def _get_by_field(
        self,
        credentials: SyftVerifyKey,
        field_name: str,
        field_value: str,
        table: Table | None = None,
    ) -> Result[Row, str]:
        table = table if table is not None else self.table
        stmt = table.select().where(
            sa.and_(
                self._get_field_filter(field_name, field_value, table=table),
                self._get_permission_filter(credentials),
            )
        )
        result = self.session.execute(stmt)
        return result

    def get_one_by_field(
        self, credentials: SyftVerifyKey, field_name: str, field_value: str
    ) -> Result[SyftT | None, str]:
        result = self._get_by_field(
            credentials=credentials, field_name=field_name, field_value=field_value
        ).first()
        if result is None:
            return Ok(None)
        return Ok(self.row_as_obj(result))

    def get_all_by_field(
        self, credentials: SyftVerifyKey, field_name: str, field_value: str
    ) -> Result[list[SyftT], str]:
        result = self._get_by_field(
            credentials=credentials, field_name=field_name, field_value=field_value
        ).all()
        objs = [self.row_as_obj(row) for row in result]
        return Ok(objs)

    def row_as_obj(self, row: Row) -> SyftT:
        return model_validate(self.object_type, row.fields)

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

    def get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool = False,
    ) -> Result[list[SyftT], str]:
        # filter by read permission
        # join on verify_key
        stmt = self.table.select()
        if not has_permission:
            stmt = stmt.where(self._get_permission_filter(credentials))
        result = self.session.execute(stmt).all()
        objs = [self.row_as_obj(row) for row in result]
        return Ok(objs)

    def update(
        self,
        credentials: SyftVerifyKey,
        obj: SyftT,
        has_permission: bool = False,
    ) -> Result[SyftT, str]:
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
            .values(fields=model_dump(obj))
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
            fields=model_dump(obj),
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
        return True
