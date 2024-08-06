# stdlib
import builtins
import contextlib
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar
from typing import get_args

# third party
from result import Err
from result import Ok
from result import Result
import sqlalchemy as sqla
from sqlalchemy import and_
from sqlalchemy import create_engine
from sqlalchemy import join
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# syft absolute
from syft.service.context import AuthedServiceContext
from syft.types.syft_object import SyftObject

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..action.action_permissions import COMPOUND_ACTION_PERMISSION
from ..response import SyftSuccess
from .job_sql import Base
from .job_sql import JobDB
from .job_sql import JobPermissionDB
from .job_sql import unwrap_uid
from .job_stash import Job
from .job_stash import JobStatus


class SQLiteDBManager:
    def __init__(self, path: Path, server_uid: UID) -> None:
        self.db_path = path / f"{server_uid}.db"
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)
        self.session_factory = sessionmaker(bind=self.engine)


ObjectT = TypeVar("ObjectT", bound=SyftObject)
SchemaT = TypeVar("SchemaT", bound=Base)


class ObjectStash(Generic[ObjectT, SchemaT]):
    object_type: ClassVar[type[ObjectT]]
    schema_type: ClassVar[type[SchemaT]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.object_type = get_args(cls.__orig_bases__[0])[0]
        cls.schema_type = get_args(cls.__orig_bases__[0])[1]

    def set(
        self,
        credentials: SyftVerifyKey,
        session: Session,
        obj: ObjectT,
    ) -> Result[ObjectT, str]:
        try:
            obj_db = self.schema_type.from_obj(obj)
            session.add(obj_db)
            session.commit()
            return Ok(obj)
        except Exception as e:
            session.rollback()
            return Err(str(e))

    def get(
        self,
        session: Session,
        uid: UID,
    ) -> Result[ObjectT, str]:
        try:
            obj_db = session.query(self.schema_type).filter_by(id=uid).first()
            if obj_db:
                return Ok(obj_db.to_obj())
            return Err(f"Object with id {uid} not found")
        except Exception as e:
            return Err(str(e))

    def delete(self, session: Session, obj_id: UID) -> Result[None, str]:
        try:
            obj_db = session.query(self.schema_type).filter_by(id=obj_id).first()
            if obj_db:
                session.delete(obj_db)
                session.commit()
                return Ok(None)
            return Err(f"Object with id {obj_id} not found")
        except Exception as e:
            session.rollback()
            return Err(str(e))

    def update(self, session: Session, obj: ObjectT) -> Result[ObjectT, str]:
        try:
            obj_db = session.query(self.schema_type).filter_by(id=obj.id).first()
            if obj_db:
                obj_db.update_obj(obj)
                session.commit()
                return Ok(obj)
            return Err(f"Object with id {obj.id} not found")
        except Exception as e:
            session.rollback()
            return Err(str(e))

    def upsert(self, session: Session, obj: ObjectT) -> Result[ObjectT, str]:
        try:
            # Use merge to handle upsert
            obj_db = session.merge(self.schema_type.from_obj(obj))
            session.commit()
            return Ok(obj_db)
        except Exception as e:
            session.rollback()
            return Err(str(e))

    def get_by_property(
        self,
        session: Session,
        property_name: str,
        property_value: Any,
    ) -> Result[ObjectT, str]:
        try:
            obj_db = (
                session.query(self.schema_type)
                .filter(getattr(self.schema_type, property_name) == property_value)
                .first()
            )
            if obj_db:
                return Ok(obj_db.to_obj())
            return Err(f"Object with {property_name} {property_value} not found")
        except Exception as e:
            return Err(str(e))

    def search(
        self,
        session: Session,
        properties: dict[str, Any],
        offset: int = 0,
        limit: int | None = None,
    ) -> Result[list[ObjectT], str]:
        # TODO check out Litestar, has example sqla crud repository with complex query support
        filters = []
        for key, value in properties.items():
            filters.append(getattr(self.schema_type, key) == value)

        try:
            query = session.query(self.schema_type).filter(*filters)
            if limit:
                query = query.limit(limit)
            query = query.offset(offset)
            return Ok([obj.to_obj() for obj in query.all()])
        except Exception as e:
            return Err(str(e))


class JobStash(ObjectStash[Job, JobDB]):
    def __init__(self) -> None:
        super().__init__(Job, JobDB)
