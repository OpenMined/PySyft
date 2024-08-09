# stdlib

# stdlib
import builtins
import contextlib
import threading
from typing import Any, ClassVar
from typing import Generic
from typing_extensions import get_args

# third party
from result import Err
from result import Ok
from result import Result
from sqlalchemy import Table
from sqlalchemy import and_
from sqlalchemy import create_engine
from sqlalchemy import or_
from sqlalchemy.orm import Query
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore, PartitionSettings
from ...types.uid import UID
from ..action.action_permissions import (
    ActionObjectEXECUTE,
    ActionObjectOWNER,
    ActionObjectPermission,
    ActionObjectREAD,
    ActionObjectWRITE,
)
from ..action.action_permissions import ActionPermission
from ..user.user_roles import ServiceRole
from .job_sql import Base, PermissionMixin
from .job_sql import JobDB
from .job_sql import ObjectT
from .job_sql import SchemaT
from .job_stash import Job
from .job_stash import JobStatus


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

    def __init__(self, store: DocumentStore) -> None:
        self.server_uid = store.server_uid
        self.verify_key = store.root_verify_key
        # self.schema_type = type(self)

        # temporary, this should be an external dependency
        self.db = SQLiteDBManager(self.server_uid)

    @property
    def session(self):
        return self.db.session

    def get(self, credentials, obj_id: UID) -> Result[ObjectT, str]:
        return self.get_one_by_property(credentials, "id", obj_id)

    @property
    def permission_cls(self):
        return self.schema_type.PermissionModel

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
                self.permission_cls.user_id == str(credentials),
                self.permission_cls.permission == permission,
            ),
            and_(
                self.permission_cls.user_id.is_(None),
                self.permission_cls.permission == compound_permission,
            ),
            and_(
                self.permission_cls.user_id.is_(None),
            ),
        )

    def check_is_admin(self, credentials: SyftVerifyKey) -> bool:
        is_admin = (
            self.session.query(
                Table(
                    "users",
                    Base.metadata,
                )
            )
            .filter_by(verify_key=str(credentials), role=ServiceRole.ADMIN)
            .first()
        )
        return is_admin

    # TODO typing
    def _get_with_permissions(
        self,
        credentials: SyftVerifyKey,
        property_name: str,
        property_value: Any,
        permission: ActionPermission,
    ) -> Query:
        is_admin = self.check_is_admin(credentials)

        base_query = self.session.query(self.schema_type)
        if is_admin:
            query = base_query
        else:
            query = base_query.join(
                self.permission_cls,
                self.schema_type.id == self.permission_cls.object_id,
            ).where(
                self._get_permissions_where(credentials, permission),
            )

        if property_name:
            query = query.filter(
                getattr(self.schema_type, property_name) == property_value
            )

        return query

    def _get_as_admin(
        self,
        property_name: str = None,
        property_value: Any = None,
    ) -> Query:
        query = self.session.query(self.schema_type)
        if property_name:
            query = query.filter(
                getattr(self.schema_type, property_name) == property_value
            )
        return query

    def get_one_as_admin(
        self,
        property_name: str,
        property_value: Any,
    ) -> Result[ObjectT, str]:
        obj = self._get_as_admin(property_name, property_value).first()
        if obj is None:
            return Ok(None)
        return Ok(obj.to_obj())

    def get_all(
        self, credentials: SyftVerifyKey, has_permission: bool = False
    ) -> Result[list[ObjectT], str]:
        is_admin = self.check_is_admin(credentials)
        if has_permission or is_admin:
            obj_dbs = self._get_as_admin(
                property_name=None,
                property_value=None,
            ).all()
        else:
            obj_dbs = self._get_with_permissions(
                credentials,
                None,
                None,
                ActionPermission.READ,
            ).all()
        return Ok([obj_db.to_obj() for obj_db in obj_dbs])

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
            return Ok(None)

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
        obj: ObjectT,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[ObjectT, str]:
        db_obj = self.schema_type.from_obj(obj)
        Permission = self.permission_cls  # type: ignore

        # TODO fix autocomplete and type checking

        db_obj.permissions = [
            Permission(
                uid=db_obj.id,
                permission=permission.permission,
                user_id=str(credentials),
            )
            for permission in self.get_ownership_permissions(obj.id, credentials)
        ]

        self.session.add(db_obj)
        self.session.commit()
        return Ok(db_obj.to_obj())

    def update(
        self,
        credentials: SyftVerifyKey,
        obj: ObjectT,
        has_permission=False,
        add_permissions=None,  # TODO: implement
    ) -> Result[ObjectT, str]:
        # _get_with_permissions checks for admin
        if has_permission:
            obj_db: SchemaT | None = self._get_as_admin(
                "id",
                obj.id,
            ).one_or_none()
        else:
            obj_db: SchemaT | None = self._get_with_permissions(
                credentials,
                "id",
                obj.id,
                ActionPermission.WRITE,
            ).one_or_none()

        if obj_db is not None:
            new_obj = self.schema_type.from_obj(obj)
            obj_db.update_obj(new_obj)
            self.session.commit()
            return Ok(obj)
        return Err(f"{self.object_type.__name__}<id={obj.id}> not found in database")

    def delete(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> Result[UID, str]:
        # TODO cascade delete permissions
        obj_db = self._get_with_permissions(
            credentials,
            "id",
            uid,
            ActionPermission.WRITE,
        ).first()

        if obj_db:
            self.session.delete(obj_db)
            self.session.commit()
            return Ok(uid)

        return Err(f"Object with id {uid} not found")

    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> Result[UID, str]:
        # TODO rename to delete
        return self.delete(credentials, uid)

    def get_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> Result[ObjectT, str]:
        obj = self.get_one_by_property(credentials, "id", uid)
        return obj

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

    def get_ownership_permissions(
        self, uid: UID, credentials: SyftVerifyKey
    ) -> list[ActionObjectPermission]:
        return [
            ActionObjectOWNER(uid=uid, credentials=credentials),
            ActionObjectWRITE(uid=uid, credentials=credentials),
            ActionObjectREAD(uid=uid, credentials=credentials),
            ActionObjectEXECUTE(uid=uid, credentials=credentials),
        ]

    def add_permissions(self, *args, **kwargs): ...
    def add_permission(self, *args, **kwargs): ...
    def add_storage_permissions(self, *args, **kwargs): ...


@serializable(canonical_name="JobStashSQL", version=1)
class JobStashSQL(ObjectStash[Job, JobDB]):
    object_type = Job
    settings: PartitionSettings = PartitionSettings(
        name=Job.__canonical_name__, object_type=Job
    )

    def __init__(self, store) -> None:
        super().__init__(store=store)

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
            result_id,
        )
        return job_db

    def get_by_parent_id(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Job | None, str]:
        subjobs = self.get_many_by_property(credentials, "parent_id", uid)
        return subjobs

    def get_active(self, credentials: SyftVerifyKey) -> Result[list[Job], str]:
        jobs = self.get_many_by_property(credentials, "status", JobStatus.CREATED)
        return jobs

    def get_by_worker(
        self, credentials: SyftVerifyKey, worker_id: str
    ) -> Result[list[Job], str]:
        jobs = self.get_many_by_property(credentials, "worker_id", worker_id)
        return jobs

    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Result[list[Job], str]:
        jobs = self.get_many_by_property(credentials, "user_code_id", user_code_id)
        return jobs

    def set(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[Job, str]:
        # relative
        from ..action.action_object import ActionObject

        # Ensure we never save cached result data in the database,
        # as they can be arbitrarily large
        if (
            isinstance(item.result, ActionObject)
            and item.result.syft_blob_storage_entry_id is not None
        ):
            item.result._clear_cache()
        obj_db = super().set(credentials, item, add_permissions, add_storage_permission)
        return obj_db

    def update(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        **kwargs,
    ) -> Result[Job, str]:
        # relative
        from ..action.action_object import ActionObject

        # Ensure we never save cached result data in the database,
        # as they can be arbitrarily large
        if (
            isinstance(item.result, ActionObject)
            and item.result.syft_blob_storage_entry_id is not None
        ):
            item.result._clear_cache()
        return super().update(credentials, item)
