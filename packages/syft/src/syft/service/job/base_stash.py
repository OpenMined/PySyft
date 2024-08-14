# stdlib

# stdlib
import base64
import json
import threading
from typing import Any
from typing import ClassVar
from typing import Generic
import uuid
from uuid import UUID

# third party
import pydantic
from result import Ok
from result import Result
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Row
from sqlalchemy import Table
from sqlalchemy import TypeDecorator
from sqlalchemy import create_engine
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
from ..action.action_object import Action
from ..action.action_permissions import ActionObjectEXECUTE
from ..action.action_permissions import ActionObjectOWNER
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionObjectREAD
from ..action.action_permissions import ActionObjectWRITE
from ..action.action_permissions import ActionPermission
from ..action.action_permissions import StoragePermission
from ..response import SyftSuccess


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


def model_dump(obj: pydantic.BaseModel) -> dict:
    obj_dict = obj.model_dump()
    for key, type_ in obj.model_fields.items():
        if type_.annotation is UID:
            obj_dict[key] = obj_dict[key].no_dash
        elif type_.annotation is SyftVerifyKey:
            obj_dict[key] = str(getattr(obj, key))
        elif type_.annotation is SyftSigningKey:
            obj_dict[key] = str(getattr(obj, key))
        elif (
            type_.annotation is LinkedObject
            or type_.annotation == Any | None  # type: ignore
            or type_.annotation == Action | None  # type: ignore
        ):
            # not very efficient as it serializes the object twice
            data = sy.serialize(getattr(obj, key), to_bytes=True)
            base64_data = base64.b64encode(data).decode("utf-8")
            obj_dict[key] = base64_data
    return obj_dict


T = TypeVar("T", bound=pydantic.BaseModel)


def model_validate(obj_type: type[T], obj_dict: dict) -> T:
    for key, type_ in obj_type.model_fields.items():
        if key not in obj_dict:
            continue
        # FIXME
        if type_.annotation is UID or type_.annotation == UID | None:
            obj_dict[key] = UID(obj_dict[key])
        elif type_.annotation is SyftVerifyKey:
            obj_dict[key] = SyftVerifyKey.from_string(obj_dict[key])
        elif type_.annotation is SyftSigningKey:
            obj_dict[key] = SyftSigningKey.from_string(obj_dict[key])
        elif (
            type_.annotation is LinkedObject
            or type_.annotation == Any | None
            or type_.annotation == Action | None
        ):
            data = base64.b64decode(obj_dict[key])
            obj_dict[key] = sy.deserialize(data, from_bytes=True)

    return obj_type.model_validate(obj_dict)


def _default_dumps(val):  # type: ignore
    if isinstance(val, UID):
        return str(val.no_dash)
    elif isinstance(val, UUID):
        return val.hex
    # raise TypeError(f"Can't serialize {val}, type {type(val)}")


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
    object_type: ClassVar[type[SyftT]]

    def __init__(self, store: DocumentStore) -> None:
        self.server_uid = store.server_uid
        self.verify_key = store.root_verify_key
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
        self, field_name: str, field_value: str
    ) -> sa.sql.elements.BinaryExpression:
        if field_name == "id":
            # use id column directly
            return self.table.c.id == field_value
        return self.table.c.fields[field_name] == field_value

    def get_one_by_field(
        self, credentials: SyftVerifyKey, field_name: str, field_value: str
    ) -> Result[SyftT | None, str]:
        result = self.session.execute(
            sa.and_(
                self._get_field_filter(field_name, field_value),
                self._get_permission_filter(credentials),
            )
        ).first()
        if result is None:
            return Ok(None)
        return Ok(self.row_as_obj(result))

    def get_all_by_field(
        self, credentials: SyftVerifyKey, field_name: str, field_value: str
    ) -> Result[list[SyftT], str]:
        stmt = self.table.select().where(
            sa.and_(
                self._get_field_filter(field_name, field_value),
                self._get_permission_filter(credentials),
            )
        )
        result = self.session.execute(stmt).all()
        objs = [self.row_as_obj(row) for row in result]
        return Ok(objs)

    def row_as_obj(self, row: Row) -> SyftT:
        return model_validate(self.object_type, row.fields)

    def _get_permission_filter(
        self,
        credentials: SyftVerifyKey,
        permission: ActionPermission = ActionPermission.READ,
    ) -> sa.sql.elements.BinaryExpression:
        # TODO: handle user.role in (ServiceRole.DATA_OWNER, ServiceRole.ADMIN)
        #       after user stash is implemented

        return self.table.c.permissions.contains(
            ActionObjectREAD(
                uid=UID(),  # dummy uid, we just need the permission string
                credentials=credentials,
            ).permission_string
        )

    def get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool = False,
    ) -> Result[list[SyftT], str]:
        # filter by read permission
        stmt = self.table.select().where(self._get_permission_filter(credentials))
        result = self.session.execute(stmt).all()
        objs = [self.row_as_obj(row) for row in result]
        return Ok(objs)

    def update(
        self,
        credentials: SyftVerifyKey,
        obj: SyftT,
        has_permission: bool = False,
    ) -> Result[SyftT, str]:
        stmt = (
            self.table.update()
            .where(
                sa.and_(
                    self._get_field_filter("id", obj.id),
                    self._get_permission_filter(credentials),
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
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,  # only used in one place, should use upsert instead
    ) -> Result[SyftT, str]:
        # uid is unique by database constraint
        uid = obj.id

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
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[SyftSuccess, str]:
        stmt = self.table.delete().where(
            sa.and_(
                self._get_field_filter("id", uid),
                self._get_permission_filter(credentials),
            )
        )
        self.session.execute(stmt)
        self.session.commit()
        return Ok(SyftSuccess())

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
