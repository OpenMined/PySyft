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
from ..action.action_permissions import ActionObjectPermission
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
            or type_.annotation == Any | None
            or type_.annotation == Action | None
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
                Column("created_at", sa.DateTime, server_default=sa.func.now()),
                Column(
                    "updated_at",
                    sa.DateTime,
                    server_default=sa.func.now(),
                    server_onupdate=sa.func.now(),
                ),
                Column("fields", JSON, default={}),
            )
        return Base.metadata.tables[table_name]

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[SyftT | None, str]:
        result = self.session.execute(
            self.table.select().where(self.table.c.id == uid)
        ).first()
        if result is None:
            return Ok(None)
        return Ok(self.row_as_obj(result))

    def get_one_by_field(
        self, credentials: SyftVerifyKey, field_name: str, field_value: str
    ) -> Result[SyftT | None, str]:
        result = self.session.execute(
            self.table.select().where(self.table.c.fields[field_name] == field_value)
        ).first()
        if result is None:
            return Ok(None)
        return Ok(self.row_as_obj(result))

    def get_all_by_field(
        self, credentials: SyftVerifyKey, field_name: str, field_value: str
    ) -> Result[list[SyftT], str]:
        result = self.session.execute(
            self.table.select().where(self.table.c.fields[field_name] == field_value)
        ).all()
        objs = [self.row_as_obj(row) for row in result]
        return Ok(objs)

    def row_as_obj(self, row: Row):
        return model_validate(self.object_type, row.fields)

    def get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool = False,
    ) -> Result[list[SyftT], str]:
        stmt = self.table.select()
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
            .where(self.table.c.id == obj.id)
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
        ignore_duplicates: bool = False,
    ) -> Result[SyftT, str]:
        stmt = self.table.insert().values(
            id=obj.id,
            fields=model_dump(obj),
        )
        self.session.execute(stmt)
        self.session.commit()
        return Ok(obj)

    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[SyftSuccess, str]:
        stmt = self.table.delete().where(self.table.c.id == uid)
        self.session.execute(stmt)
        self.session.commit()
        return Ok(SyftSuccess())

    def add_permissions(self, permissions: list[ActionObjectPermission]) -> None:
        pass

    def add_permission(self, permission: ActionObjectPermission) -> None:
        pass

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        pass

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        return True

    def has_storage_permission(self, permission) -> bool:
        return True
