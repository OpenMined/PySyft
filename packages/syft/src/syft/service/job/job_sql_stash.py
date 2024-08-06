# stdlib

# stdlib
import builtins
import contextlib
from re import I
import threading
from typing import Any, ClassVar, Generic, get_args
from syft.types.syft_object import SyftObject
from typing_extensions import TypeVar

# third party
from result import Ok, Err
from result import Result
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ..action.action_object import ActionObject
from ..action.action_permissions import (
    COMPOUND_ACTION_PERMISSION,
    ActionObjectPermission,
)
from ..action.action_permissions import ActionPermission
from ..response import SyftSuccess
from .job_sql import Base, JobPermissionDB, ObjectT, SchemaT
from .job_sql import JobDB
from .job_sql import unwrap_uid
from .job_stash import Job
from .job_stash import JobStatus
from sqlalchemy.orm import Session
from sqlalchemy import select, or_, and_, join
from sqlalchemy.orm import Query


def get_permission_where(user_id):
    return or_(
        JobPermissionDB.user_id == user_id,
        and_(
            JobPermissionDB.user_id is None,
            JobPermissionDB.permission.in_(
                COMPOUND_ACTION_PERMISSION,
            ),
        ),
    )


class SQLiteDBManager:
    def __init__(self, server_uid) -> None:
        self.server_uid = server_uid
        self.path = f"sqlite:////tmp/{server_uid}.db"
        self.engine = create_engine(self.path)
        print(f"Connecting to {self.path}")
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.thread_local = threading.local()

        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        if not hasattr(self.thread_local, "session"):
            self.thread_local.session = self.SessionFactory()
        return self.thread_local.session

    @contextlib.contextmanager
    def session_context(self):
        session = self.get_session()
        yield session

    @property
    def session(self):
        return self.get_session()


class ObjectStash(Generic[ObjectT, SchemaT]):
    object_type: ClassVar[type[ObjectT]]
    schema_type: ClassVar[type[SchemaT]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.object_type = get_args(cls.__orig_bases__[0])[0]
        cls.schema_type = get_args(cls.__orig_bases__[0])[1]

    def __init__(self, server_uid: str) -> None:
        self.server_uid = server_uid
        self.schema_type = SchemaT

        # temporary, this should be an external dependency
        self.db = SQLiteDBManager(server_uid)

    @property
    def session(self):
        return self.db.session

    def get(self, credentials, obj_id: UID) -> Result[ObjectT, str]:
        return self.get_one_by_property(credentials, "id", unwrap_uid(obj_id))

    @property
    def permission_cls(self):
        return self.schema_type.Permission

    def _get_permissions_where(
        self, credentials: SyftVerifyKey, permission: ActionPermission
    ) -> Any:
        if permission == ActionPermission.READ:
            compound_permission = ActionPermission.ALL_READ
        elif permission == ActionPermission.WRITE:
            compound_permission = ActionPermission.ALL_WRITE
        else:
            raise ValueError(f"Permission type {permission} not supported")

        return or_(
            and_(
                self.permission_cls.user_id == str(credentials.verify_key),
                self.permission_cls.permission == permission,
            ),
            and_(
                self.permission_cls.user_id is None,
                self.permission_cls.permission == compound_permission,
            ),
        )

    # TODO typing
    def _get_with_permissions(
        self,
        credentials: SyftVerifyKey,
        property_name: str,
        property_value: Any,
        permission: ActionPermission,
    ) -> Query:
        query = (
            self.session.query(self.schema_type)
            .join(
                self.permission_cls,
                self.schema_type.id == self.permission_cls.object_id,
            )
            .where(
                self._get_permissions_where(credentials, permission),
            )
        )

        if property_name:
            query = query.filter(
                getattr(self.schema_type, property_name) == property_value
            )

        return query

    def get_one_by_property(
        self,
        credentials: SyftVerifyKey,
        property_name: str,
        property_value: Any,
    ) -> Result[ObjectT, str]:
        try:
            obj_db: SchemaT | None = self._get_with_permissions(
                credentials,
                property_name,
                property_value,
                ActionPermission.READ,
            ).first()

            if obj_db is not None:
                return Ok(obj_db.to_obj())
            return Err(f"Object with {property_name} {property_value} not found")

        except Exception as e:
            return Err(str(e))

    def get_many_by_property(
        self,
        credentials: SyftVerifyKey,
        property_name: str,
        property_value: Any,
    ) -> Result[list[ObjectT], str]:
        try:
            obj_dbs = self._get_with_permissions(
                credentials,
                property_name,
                property_value,
                ActionPermission.READ,
            ).all()
            return Ok([obj_db.to_obj() for obj_db in obj_dbs])
        except Exception as e:
            return Err(str(e))

    def get_index(
        self,
        credentials: SyftVerifyKey,
        index: int,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> Result[ObjectT, str]:
        obj: SchemaT | None = (
            self.session.query(self.schema_type)
            .join(
                self.permission_cls,
                self.schema_type.id == self.permission_cls.object_id,
            )
            .where(
                self._get_permissions_where(credentials, ActionPermission.READ),
            )
            .order_by(
                getattr(self.schema_type, order_by).desc() if descending else None
            )
            .offset(index)
            .first()
        )
        if obj is None:
            return Err("Object not found")
        return Ok(obj.to_obj())

    def set(
        self,
        credentials: SyftVerifyKey,
        item: ObjectT,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[ObjectT, str]:
        try:
            db_obj = self.schema_type.from_obj(item)
            Permission = self.schema_type.Permission

            # TODO fix autocomplete and type checking
            db_obj.permissions = {
                Permission(
                    id=db_obj.id,
                    permission=ActionPermission.READ,
                    user_id=str(credentials.verify_key),
                )
            }
            if add_permissions:
                db_obj.permissions |= {
                    Permission(
                        uid=db_obj.id,
                        permission=permission.permission,
                        user_id=str(permission.credentials)
                        if permission.credentials
                        else None,
                    )
                    for permission in add_permissions
                }

            self.session.add(db_obj)
            self.session.commit()
            return Ok(db_obj.to_obj())
        except Exception as e:
            self.session.rollback()
            return Err(str(e))

    def update(self, credentials: SyftVerifyKey, obj: ObjectT) -> Result[ObjectT, str]:
        try:
            obj_db = self._get_with_permissions(
                credentials,
                "id",
                obj.id,
                ActionPermission.WRITE,
            ).first()

            if obj_db:
                obj_db.update_obj(obj)
                self.session.commit()
                return Ok(obj)
            return Err(f"Object with id {obj.id} not found")
        except Exception as e:
            self.session.rollback()
            return Err(str(e))

    def delete(self, credentials: SyftVerifyKey, uid: UID) -> Result[UID, str]:
        try:
            obj_db = self._get_with_permissions(
                credentials,
                "id",
                unwrap_uid(uid),
                ActionPermission.WRITE,
            ).first()

            if obj_db:
                self.session.delete(obj_db)
                self.session.commit()
                return Ok(uid)

            return Err(f"Object with id {uid} not found")

        except Exception as e:
            return Err(str(e))


@serializable(canonical_name="JobStashSQL", version=1)
class JobStashSQL(ObjectStash[Job, JobDB]):
    object_type = Job
    settings: PartitionSettings = PartitionSettings(
        name=Job.__canonical_name__, object_type=Job
    )

    def __init__(self, server_uid: str) -> None:
        super().__init__(server_uid)
        self.schema_type = JobDB

    def set_result(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        add_permissions: list[ActionObjectPermission] | None = None,
    ) -> Result[Job | None, str]:
        return self.update(credentials, item)

    def get_by_result_id(
        self,
        credentials: SyftVerifyKey,
        result_id: UID,
    ) -> Result[Job | None, str]:
        job_db = self.get_one_by_property(
            credentials,
            "result_id",
            unwrap_uid(result_id),
        )
        return job_db

    def get_by_parent_id(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Job | None, str]:
        subjobs = self.get_many_by_property(credentials, "parent_id", unwrap_uid(uid))
        return subjobs

    def delete_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> Result[Ok, str]:
        with self.session_context() as session:
            try:
                session.query(JobDB).filter_by(id=unwrap_uid(uid)).delete()
                session.commit()
                return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
            except IntegrityError as e:
                return str(e)

    def get_active(self, credentials: SyftVerifyKey) -> Result[list[Job], str]:
        jobs = self.get_many_by_property(credentials, "status", JobStatus.CREATED)
        return jobs

    def get_by_worker(
        self, credentials: SyftVerifyKey, worker_id: str
    ) -> Result[list[Job], str]:
        with self.session_context() as session:
            jobs = (
                session.query(JobDB)
                .join(JobPermissionDB, JobDB.id == JobPermissionDB.object_id)
                .where(get_permission_where(str(credentials.verify_key)))
                .filter_by(worker_id=worker_id)
                .all()
            )
            return Ok([job.to_obj() for job in jobs])

    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Result[list[Job], str]:
        with self.session_context() as session:
            jobs = (
                session.query(JobDB)
                .join(JobPermissionDB, JobDB.id == JobPermissionDB.object_id)
                .where(get_permission_where(str(credentials.verify_key)))
                .filter_by(user_code_id=unwrap_uid(user_code_id))
                .all()
            )
            return Ok([job.to_obj() for job in jobs])

    def get_all(self, credentials: SyftVerifyKey) -> Result[list[Job], str]:
        user_id = str(credentials.verify_key)

        with self.session_context() as session:
            jobs = session.execute(
                select(JobDB)
                .join(JobPermissionDB, JobDB.id == JobPermissionDB.object_id)
                .where(get_permission_where(str(credentials.verify_key)))
            ).scalars()
            return Ok([job.to_obj() for job in jobs])

    def set(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[Job, str]:
        # Ensure we never save cached result data in the database,
        # as they can be arbitrarily large
        if (
            isinstance(item.result, ActionObject)
            and item.result.syft_blob_storage_entry_id is not None
        ):
            item.result._clear_cache()
        obj_db = super().set(credentials, item, add_permissions, add_storage_permission)
        return obj_db

    def get_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> Result[Job, str]:
        job_db = self.get_one_by_property(credentials, "id", unwrap_uid(uid))
        return job_db

    def update(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        **kwargs,
    ) -> Result[Job, str]:
        # Ensure we never save cached result data in the database,
        # as they can be arbitrarily large
        if (
            isinstance(item.result, ActionObject)
            and item.result.syft_blob_storage_entry_id is not None
        ):
            item.result._clear_cache()
        with self.session_context() as session:
            # TODO we need to check write permissions here
            updated_job_db = JobDB.from_obj(item)
            job_db = (
                session.query(JobDB)
                .join(JobPermissionDB, JobDB.id == JobPermissionDB.object_id)
                .where(get_permission_where(str(credentials.verify_key)))
                .filter_by(id=unwrap_uid(item.id))
                .first()
            )

            if job_db is None:
                return Err("Job not found")

            for key, value in updated_job_db.to_dict().items():
                setattr(job_db, key, value)

            # session.add(job_db)
            session.commit()
            return Ok(job_db.to_obj())

    def has_permission(self, *args, **kwargs) -> bool:
        return True

    def _get_permissions_for_uid(self, uid: UID) -> Result[builtins.set[str], str]:
        return Ok(
            {
                ActionPermission.ALL_READ.name,
                ActionPermission.ALL_WRITE.name,
                ActionPermission.ALL_EXECUTE.name,
            }
        )

    def _get_storage_permissions_for_uid(
        self, uid: UID
    ) -> Result[builtins.set[str], str]:
        return Ok(
            {
                self.server_uid,
            }
        )

    def add_permission(self, *args, **kwargs): ...
    def add_storage_permissions(self, *args, **kwargs): ...
