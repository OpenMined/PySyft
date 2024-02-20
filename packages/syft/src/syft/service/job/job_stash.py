# stdlib
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
import pydantic
from result import Err
from result import Ok
from result import Result

# relative
from ...client.api import APIRegistry
from ...client.api import SyftAPICall
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...types.datetime import DateTime
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SyftObject
from ...types.syft_object import short_uid
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.markdown import as_markdown_code
from ...util.telemetry import instrument
from ..action.action_data_empty import ActionDataLink
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..response import SyftError
from ..response import SyftNotReady
from ..response import SyftSuccess


@serializable()
class JobStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    ERRORED = "errored"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


@serializable()
class JobV1(SyftObject):
    __canonical_name__ = "JobItem"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: UID
    result: Optional[Any]
    resolved: bool = False
    status: JobStatus = JobStatus.CREATED
    log_id: Optional[UID]
    parent_job_id: Optional[UID]
    n_iters: Optional[int] = 0
    current_iter: Optional[int] = None
    creation_time: Optional[str] = None
    action: Optional[Action] = None


@serializable()
class JobV2(SyftObject):
    __canonical_name__ = "JobItem"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    node_uid: UID
    result: Optional[Any]
    resolved: bool = False
    status: JobStatus = JobStatus.CREATED
    log_id: Optional[UID]
    parent_job_id: Optional[UID]
    n_iters: Optional[int] = 0
    current_iter: Optional[int] = None
    creation_time: Optional[str] = None
    action: Optional[Action] = None
    job_pid: Optional[int] = None


@serializable()
class Job(SyftObject):
    __canonical_name__ = "JobItem"
    __version__ = SYFT_OBJECT_VERSION_3

    id: UID
    node_uid: UID
    result: Optional[Any]
    resolved: bool = False
    status: JobStatus = JobStatus.CREATED
    log_id: Optional[UID]
    parent_job_id: Optional[UID]
    n_iters: Optional[int] = 0
    current_iter: Optional[int] = None
    creation_time: Optional[str] = None
    action: Optional[Action] = None
    job_pid: Optional[int] = None
    job_worker_id: Optional[UID] = None
    updated_at: Optional[DateTime] = None
    user_code_id: Optional[UID] = None

    __attr_searchable__ = ["parent_job_id", "job_worker_id", "status", "user_code_id"]
    __repr_attrs__ = ["id", "result", "resolved", "progress", "creation_time"]

    @pydantic.root_validator()
    def check_time(cls, values: dict) -> dict:
        if values.get("creation_time", None) is None:
            values["creation_time"] = str(datetime.now())
        return values

    @pydantic.root_validator()
    def check_user_code_id(cls, values: dict) -> dict:
        action = values.get("action")
        user_code_id = values.get("user_code_id")

        if action is not None:
            if user_code_id is None:
                values["user_code_id"] = action.user_code_id
            elif action.user_code_id != user_code_id:
                raise pydantic.ValidationError(
                    "user_code_id does not match the action's user_code_id", cls
                )

        return values

    @property
    def action_display_name(self):
        if self.action is None:
            return "action"
        else:
            # hacky
            self.action.syft_node_location = self.syft_node_location
            self.action.syft_client_verify_key = self.syft_client_verify_key
            return self.action.job_display_name

    @property
    def time_remaining_string(self):
        # update state
        self.fetch()
        percentage = round((self.current_iter / self.n_iters) * 100)
        blocks_filled = round(percentage / 20)
        blocks_empty = 5 - blocks_filled
        blocks_filled_str = "█" * blocks_filled
        blocks_empty_str = "&nbsp;&nbsp;" * blocks_empty
        return f"{percentage}% |{blocks_filled_str}{blocks_empty_str}|\n{self.current_iter}/{self.n_iters}\n"

    @property
    def worker(self):
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.worker.get(self.job_worker_id)

    @property
    def eta_string(self):
        if (
            self.current_iter is None
            or self.current_iter == 0
            or self.n_iters is None
            or self.creation_time is None
        ):
            return None

        def format_timedelta(local_timedelta):
            total_seconds = int(local_timedelta.total_seconds())
            hours, leftover = divmod(total_seconds, 3600)
            minutes, seconds = divmod(leftover, 60)

            hours_string = f"{hours}:" if hours != 0 else ""
            minutes_string = f"{minutes}:".zfill(3)
            seconds_string = f"{seconds}".zfill(2)

            return f"{hours_string}{minutes_string}{seconds_string}"

        now = datetime.now()
        time_passed = now - datetime.fromisoformat(self.creation_time)
        iter_duration_seconds: float = time_passed.total_seconds() / self.current_iter
        iters_remaining = self.n_iters - self.current_iter

        # TODO: Adjust by the number of consumers
        time_remaining = timedelta(seconds=iters_remaining * iter_duration_seconds)
        time_passed_str = format_timedelta(time_passed)
        time_remaining_str = format_timedelta(time_remaining)

        if iter_duration_seconds >= 1:
            iter_duration: timedelta = timedelta(seconds=iter_duration_seconds)
            iter_duration_str = f"{format_timedelta(iter_duration)}s/it"
        else:
            iters_per_second = round(1 / iter_duration_seconds)
            iter_duration_str = f"{iters_per_second}it/s"

        return f"[{time_passed_str}<{time_remaining_str}]\n{iter_duration_str}"

    @property
    def progress(self) -> str:
        if self.status in [JobStatus.PROCESSING, JobStatus.COMPLETED]:
            if self.current_iter is None:
                return ""
            else:
                if self.n_iters is not None:
                    return self.time_remaining_string
                # if self.current_iter !=0
                # we can compute the remaining time

                # we cannot compute the remaining time
                else:
                    n_iters_str = "?" if self.n_iters is None else str(self.n_iters)
                    return f"{self.current_iter}/{n_iters_str}"
        else:
            return ""

    def info(
        self,
        public_metadata: bool = True,
        result: bool = False,
    ) -> "JobInfo":
        return JobInfo.from_job(self, public_metadata, result)

    def apply_info(self, info: "JobInfo") -> None:
        if info.includes_metadata:
            for attr in info.__public_metadata_attrs__:
                setattr(self, attr, getattr(info, attr))

        if info.includes_result:
            self.result = info.result

    def restart(self, kill=False) -> None:
        if kill:
            self.kill()
        self.fetch()
        if not self.has_parent:
            # this is currently the limitation, we will need to implement
            # killing toplevel jobs later
            print("Can only kill nested jobs")
        elif kill or (
            self.status != JobStatus.PROCESSING and self.status != JobStatus.CREATED
        ):
            api = APIRegistry.api_for(
                node_uid=self.syft_node_location,
                user_verify_key=self.syft_client_verify_key,
            )
            call = SyftAPICall(
                node_uid=self.node_uid,
                path="job.restart",
                args=[],
                kwargs={"uid": self.id},
                blocking=True,
            )

            api.make_call(call)
        else:
            print(
                "Job is running or scheduled, if you want to kill it use job.kill() first"
            )

    def kill(self) -> Union[None, SyftError]:
        if self.job_pid is not None:
            api = APIRegistry.api_for(
                node_uid=self.syft_node_location,
                user_verify_key=self.syft_client_verify_key,
            )

            call = SyftAPICall(
                node_uid=self.node_uid,
                path="job.kill",
                args=[],
                kwargs={"id": self.id},
                blocking=True,
            )
            api.make_call(call)
        else:
            return SyftError(
                message="Job is not running or isn't running in multiprocessing mode."
            )

    def fetch(self) -> None:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        call = SyftAPICall(
            node_uid=self.node_uid,
            path="job.get",
            args=[],
            kwargs={"uid": self.id},
            blocking=True,
        )
        job = api.make_call(call)
        self.resolved = job.resolved
        if job.resolved:
            self.result = job.result

        self.status = job.status
        self.n_iters = job.n_iters
        self.current_iter = job.current_iter

    @property
    def subjobs(self):
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.job.get_subjobs(self.id)

    @property
    def owner(self):
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.user.get_current_user(self.id)

    def logs(self, stdout=True, stderr=True, _print=True):
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        results = []
        if stdout:
            stdout_log = api.services.log.get(self.log_id)
            results.append(stdout_log)

        if stderr:
            try:
                std_err_log = api.services.log.get_error(self.log_id)
                results.append(std_err_log)
            except Exception:
                # no access
                if isinstance(self.result, Err):
                    results.append(self.result.value)
        else:
            # add short error
            if isinstance(self.result, Err):
                results.append(self.result.value)

        results_str = "\n".join(results)
        if not _print:
            return results_str
        else:
            print(results_str)

    # def __repr__(self) -> str:
    #     return f"<Job: {self.id}>: {self.status}"

    def _coll_repr_(self) -> Dict[str, Any]:
        logs = self.logs(_print=False, stderr=False)
        log_lines = logs.split("\n")
        subjobs = self.subjobs
        if len(log_lines) > 2:
            logs = f"... ({len(log_lines)} lines)\n" + "\n".join(log_lines[-2:])
        else:
            logs = logs

        return {
            "status": f"{self.action_display_name}: {self.status}"
            + (
                f"\non worker {short_uid(self.job_worker_id)}"
                if self.job_worker_id
                else ""
            ),
            "progress": self.progress,
            "eta": self.eta_string,
            "created": f"{self.creation_time[:-7]} by {self.owner.email}",
            "logs": logs,
            # "result": result,
            # "parent_id": str(self.parent_job_id) if self.parent_job_id else "-",
            "subjobs": len(subjobs),
        }

    @property
    def has_parent(self):
        return self.parent_job_id is not None

    def _repr_markdown_(self) -> str:
        _ = self.resolve
        logs = self.logs(_print=False)
        logs_w_linenr = "\n".join(
            [f"{i} {line}" for i, line in enumerate(logs.rstrip().split("\n"))]
        )

        if self.status == JobStatus.COMPLETED:
            logs_w_linenr += "\nJOB COMPLETED"

        md = f"""class Job:
    id: UID = {self.id}
    status: {self.status}
    has_parent: {self.has_parent}
    result: {self.result.__str__()}
    logs:

{logs_w_linenr}
    """
        return as_markdown_code(md)

    def wait(self, job_only=False):
        # stdlib
        from time import sleep

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )

        # todo: timeout
        if self.resolved:
            return self.resolve

        if not job_only:
            self.result.wait()

        print_warning = True
        while True:
            self.fetch()
            if print_warning:
                result_obj = api.services.action.get(
                    self.result.id, resolve_nested=False
                )
                if isinstance(result_obj.syft_action_data, ActionDataLink) and job_only:
                    print(
                        "You're trying to wait on a job that has a link as a result."
                        "This means that the job may be ready but the linked result may not."
                        "Use job.wait().get() instead to wait for the linked result."
                    )
                    print_warning = False
            sleep(2)
            if self.resolved:
                break
        return self.resolve

    @property
    def resolve(self) -> Union[Any, SyftNotReady]:
        if not self.resolved:
            self.fetch()

        if self.resolved:
            return self.result
        return SyftNotReady(message=f"{self.id} not ready yet.")


@serializable()
class JobInfo(SyftObject):
    __canonical_name__ = "JobInfo"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = [
        "resolved",
        "status",
        "n_iters",
        "current_iter",
        "creation_time",
    ]
    __public_metadata_attrs__ = [
        "resolved",
        "status",
        "n_iters",
        "current_iter",
        "creation_time",
    ]
    # Separate check if the job has logs, result, or metadata
    # None check is not enough because the values we set could be None
    includes_metadata: bool
    includes_result: bool
    # TODO add logs (error reporting PRD)

    resolved: Optional[bool] = None
    status: Optional[JobStatus] = None
    n_iters: Optional[int] = None
    current_iter: Optional[int] = None
    creation_time: Optional[str] = None

    result: Optional[Any] = None

    def _repr_html_(self) -> str:
        metadata_str = ""
        if self.includes_metadata:
            metadata_str += "<h4>Public metadata</h4>"
            for attr in self.__public_metadata_attrs__:
                value = getattr(self, attr, None)
                if value is not None:
                    metadata_str += f"<p style='margin-left: 10px;'><strong>{attr}:</strong> {value}</p>"

        result_str = "<h4>Result</h4>"
        if self.includes_result:
            result_str += f"<p style='margin-left: 10px;'>{str(self.result)}</p>"
        else:
            result_str += "<p style='margin-left: 10px;'><i>No result included</i></p>"

        return f"""
            <style>
            .job-info {{color: {SURFACE[options.color_theme]};}}
            </style>
            <div class='job-info'>
                <h3>JobInfo</h3>
                {metadata_str}
                {result_str}
            </div>
        """

    @classmethod
    def from_job(
        cls,
        job: Job,
        metadata: bool = False,
        result: bool = False,
    ):
        info = cls(
            includes_metadata=metadata,
            includes_result=result,
        )

        if metadata:
            for attr in cls.__public_metadata_attrs__:
                setattr(info, attr, getattr(job, attr))

        if result:
            if not job.resolved:
                raise ValueError("Cannot sync result of unresolved job")
            if not isinstance(job.result, ActionObject):
                raise ValueError("Could not sync result of job")
            info.result = job.result.get()

        return info


@migrate(Job, JobV2)
def downgrade_job_v3_to_v2():
    return [drop(["job_worker_id", "user_code_id"])]


@migrate(JobV2, Job)
def upgrade_job_v2_to_v3():
    return [
        make_set_default("job_worker_id", None),
        make_set_default("user_code_id", None),
    ]


@migrate(JobV2, JobV1)
def downgrade_job_v2_to_v1():
    return [
        drop("job_pid"),
    ]


@migrate(JobV1, JobV2)
def upgrade_job_v1_to_v2():
    return [make_set_default("job_pid", None)]


@instrument
@serializable()
class JobStash(BaseStash):
    object_type = Job
    settings: PartitionSettings = PartitionSettings(
        name=Job.__canonical_name__, object_type=Job
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set_result(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
    ) -> Result[Optional[Job], str]:
        valid = self.check_type(item, self.object_type)
        if valid.is_err():
            return SyftError(message=valid.err())
        return super().update(credentials, item, add_permissions)

    def set_placeholder(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
    ) -> Result[Job, str]:
        # 🟡 TODO 36: Needs distributed lock
        if not item.resolved:
            exists = self.get_by_uid(credentials, item.id)
            if exists.is_ok() and exists.ok() is None:
                valid = self.check_type(item, self.object_type)
                if valid.is_err():
                    return SyftError(message=valid.err())
                return super().set(credentials, item, add_permissions)
        return item

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Optional[Job], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        item = self.query_one(credentials=credentials, qks=qks)
        return item

    def get_by_parent_id(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Optional[Job], str]:
        qks = QueryKeys(
            qks=[PartitionKey(key="parent_job_id", type_=UID).with_obj(uid)]
        )
        item = self.query_all(credentials=credentials, qks=qks)
        return item

    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[SyftSuccess, str]:
        qk = UIDPartitionKey.with_obj(uid)
        result = super().delete(credentials=credentials, qk=qk)
        if result.is_ok():
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return result

    def get_active(self, credentials: SyftVerifyKey) -> Result[SyftSuccess, str]:
        qks = QueryKeys(
            qks=[
                PartitionKey(key="status", type_=JobStatus).with_obj(
                    JobStatus.PROCESSING
                )
            ]
        )
        return self.query_all(credentials=credentials, qks=qks)

    def get_by_worker(self, credentials: SyftVerifyKey, worker_id: str):
        qks = QueryKeys(
            qks=[PartitionKey(key="job_worker_id", type_=str).with_obj(worker_id)]
        )
        return self.query_all(credentials=credentials, qks=qks)

    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Union[List[Job], SyftError]:
        qks = QueryKeys(
            qks=[PartitionKey(key="user_code_id", type_=UID).with_obj(user_code_id)]
        )

        return self.query_all(credentials=credentials, qks=qks)
