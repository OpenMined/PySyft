# stdlib

# stdlib
import builtins
import contextlib
import threading

# third party
from result import Ok
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
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..response import SyftSuccess
from .job_sql import Base
from .job_sql import JobDB
from .job_sql import unwrap_uid
from .job_stash import Job
from .job_stash import JobStatus


@serializable(canonical_name="JobStashSQL", version=1)
class JobStashSQL:
    object_type = Job
    settings: PartitionSettings = PartitionSettings(
        name=Job.__canonical_name__, object_type=Job
    )

    def __init__(self, server_uid) -> None:
        self.server_uid = server_uid
        self.engine = create_engine(f"sqlite:////tmp/{server_uid}.db")
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
        with self.session_context() as session:
            job_db = (
                session.query(JobDB).filter_by(result_id=unwrap_uid(result_id)).first()
            )
            if job_db is None:
                return Ok(None)
            return Ok(job_db.to_obj())

    def get_by_parent_id(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Job | None, str]:
        with self.session_context() as session:
            subjobs = session.query(JobDB).filter_by(parent_id=unwrap_uid(uid)).all()
            return Ok([job.to_obj() for job in subjobs])

    def delete_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> Result[Ok, str]:
        with self.session_context() as session:
            try:
                session.query(JobDB).filter_by(id=unwrap_uid(uid)).delete()
                session.commit()
                return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
            except IntegrityError as e:
                return str(e)

    def get_active(self, credentials: SyftVerifyKey) -> Result[list[Job], str]:
        with self.session_context() as session:
            jobs = session.query(JobDB).filter_by(status=JobStatus.PROCESSING).all()
            return Ok([job.to_obj() for job in jobs])

    def get_by_worker(
        self, credentials: SyftVerifyKey, worker_id: str
    ) -> Result[list[Job], str]:
        with self.session_context() as session:
            jobs = session.query(JobDB).filter_by(worker_id=unwrap_uid(worker_id)).all()
            return Ok([job.to_obj() for job in jobs])

    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Result[list[Job], str]:
        with self.session_context() as session:
            jobs = (
                session.query(JobDB)
                .filter_by(user_code_id=unwrap_uid(user_code_id))
                .all()
            )
            return Ok([job.to_obj() for job in jobs])

    def get_all(self, credentials: SyftVerifyKey) -> Result[list[Job], str]:
        with self.session_context() as session:
            jobs = session.query(JobDB).all()
            return Ok([job.to_obj() for job in jobs])

    def set(
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
        job_db = JobDB.from_obj(item)

        with self.session_context() as session:
            session.add(job_db)
            session.flush()
            session.commit()
            return Ok(job_db.to_obj())

    def get_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> Result[Job, str]:
        with self.session_context() as session:
            job_db = session.query(JobDB).filter_by(id=unwrap_uid(uid)).first()
            if job_db is None:
                return Ok(None)
            return Ok(job_db.to_obj())

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
            job_db = JobDB.from_obj(item)
            session.query(JobDB).filter_by(id=unwrap_uid(item.id)).update(
                job_db.to_dict()
            )
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
