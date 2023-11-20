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
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.markdown import as_markdown_code
from ...util.telemetry import instrument
from ..action.action_object import Action
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


@serializable()
class Job(SyftObject):
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

    __attr_searchable__ = ["parent_job_id"]
    __repr_attrs__ = ["id", "result", "resolved", "progress", "creation_time"]

    @pydantic.root_validator()
    def check_time(cls, values: dict) -> dict:
        if values.get("creation_time", None) is None:
            values["creation_time"] = str(datetime.now())
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

        iter_duration_seconds = time_passed.total_seconds() / self.current_iter
        iter_duration = timedelta(seconds=iter_duration_seconds)

        iters_remaining = self.n_iters - self.current_iter
        # TODO: Adjust by the number of consumers
        time_remaining = iters_remaining * iter_duration

        time_passed_str = format_timedelta(time_passed)
        time_remaining_str = format_timedelta(time_remaining)
        iter_duration_str = format_timedelta(iter_duration)

        return f"[{time_passed_str}<{time_remaining_str}]\n{iter_duration_str}s/it"

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

    def restart(self) -> None:
        if self.status != JobStatus.PROCESSING and self.status != JobStatus.CREATED:
            api = APIRegistry.api_for(
                node_uid=self.node_uid,
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
            print("Job is already running or in the queue.")

    def fetch(self) -> None:
        api = APIRegistry.api_for(
            node_uid=self.node_uid,
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
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.job.get_subjobs(self.id)

    @property
    def owner(self):
        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.user.get_current_user(self.id)

    def logs(self, stdout=True, stderr=True, _print=True):
        api = APIRegistry.api_for(
            node_uid=self.node_uid,
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
            "status": f"{self.action_display_name}: {self.status}",
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

    def wait(self):
        # stdlib
        from time import sleep

        # todo: timeout
        if self.resolved:
            return self.resolve
        while True:
            self.fetch()
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
