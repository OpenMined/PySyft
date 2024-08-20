# stdlib
from collections.abc import Callable
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
import random
from string import Template
from time import sleep
from typing import Any

# third party
from pydantic import Field
from pydantic import model_validator
from result import Err
from result import Result
from typing_extensions import Self

# relative
from ...client.api import APIRegistry
from ...client.api import SyftAPICall
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...service.context import AuthedServiceContext
from ...service.worker.worker_pool import SyftWorker
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...types.datetime import DateTime
from ...types.datetime import format_timedelta
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.transforms import make_set_default
from ...types.uid import UID
from ...util.markdown import as_markdown_code
from ...util.telemetry import instrument
from ...util.util import prompt_warning_message
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..log.log import SyftLog
from ..response import SyftError
from ..response import SyftNotReady
from ..response import SyftSuccess
from ..user.user import UserView
from .base_stash import ObjectStash
from .html_template import job_repr_template


@serializable(canonical_name="JobStatus", version=1)
class JobStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    ERRORED = "errored"
    COMPLETED = "completed"
    TERMINATING = "terminating"
    INTERRUPTED = "interrupted"


def center_content(text: Any) -> str:
    if isinstance(text, str):
        text = text.replace("\n", "<br>")
    center_div = f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center; width: 100%; height: 100%;">
        {text}
    </div>
    """
    center_div = center_div.replace("\n", "")
    return center_div


@serializable(canonical_name="JobType", version=1)
class JobType(str, Enum):
    JOB = "job"
    TWINAPIJOB = "twinapijob"

    def __str__(self) -> str:
        return self.value


@serializable()
class Job(SyncableSyftObject):
    __canonical_name__ = "JobItem"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    server_uid: UID
    result: Any | None = None
    resolved: bool = False
    status: JobStatus = JobStatus.CREATED
    log_id: UID | None = None
    parent_job_id: UID | None = None
    n_iters: int | None = 0
    current_iter: int | None = None
    creation_time: str | None = Field(
        default_factory=lambda: str(datetime.now(tz=timezone.utc))
    )
    action: Action | None = None
    job_pid: int | None = None
    job_worker_id: UID | None = None
    updated_at: DateTime | None = None
    user_code_id: UID | None = None
    requested_by: UID | None = None
    job_type: JobType = JobType.JOB
    # used by JobType.TWINAPIJOB
    endpoint: str | None = None

    __attr_searchable__ = [
        "parent_job_id",
        "job_worker_id",
        "status",
        "user_code_id",
        "result_id",
    ]
    __repr_attrs__ = [
        "id",
        "result",
        "resolved",
        "progress",
        "creation_time",
        "user_code_name",
    ]
    __exclude_sync_diff_attrs__ = ["action", "server_uid"]
    __table_coll_widths__ = [
        "min-content",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
    ]
    __syft_include_id_coll_repr__ = False

    @model_validator(mode="after")
    def check_user_code_id(self) -> Self:
        if self.action is not None:
            if self.user_code_id is None:
                self.user_code_id = self.action.user_code_id
            elif self.action.user_code_id != self.user_code_id:
                raise ValueError(
                    "user_code_id does not match the action's user_code_id",
                    self.__class__,
                )

        return self

    @property
    def result_id(self) -> UID | None:
        if isinstance(self.result, ActionObject):
            return self.result.id.id
        return None

    @property
    def action_display_name(self) -> str:
        if self.action is None:
            return "action"
        else:
            # hacky
            self.action.syft_server_location = self.syft_server_location
            self.action.syft_client_verify_key = self.syft_client_verify_key
            return self.action.job_display_name

    @property
    def user_code_name(self) -> str | None:
        if self.user_code_id is not None:
            api = APIRegistry.api_for(
                server_uid=self.syft_server_location,
                user_verify_key=self.syft_client_verify_key,
            )
            if api is None:
                return None
            user_code = api.services.code.get_by_id(self.user_code_id)
            return user_code.service_func_name
        return None

    @property
    def time_remaining_string(self) -> str | None:
        # update state
        self.fetch()
        if (
            self.current_iter is not None
            and self.n_iters is not None
            and self.n_iters != 0
        ):
            percentage = round((self.current_iter / self.n_iters) * 100)
            blocks_filled = round(percentage / 20)
            blocks_empty = 5 - blocks_filled
            blocks_filled_str = "█" * blocks_filled
            blocks_empty_str = "&nbsp;&nbsp;" * blocks_empty
            return f"{percentage}% |{blocks_filled_str}{blocks_empty_str}|\n{self.current_iter}/{self.n_iters}\n"
        return None

    @property
    def worker(self) -> SyftWorker | SyftError:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        return api.services.worker.get(self.job_worker_id)

    @property
    def eta_string(self) -> str | None:
        if (
            self.current_iter is None
            or self.current_iter == 0
            or self.n_iters is None
            or self.creation_time is None
        ):
            return None

        now = datetime.now(tz=timezone.utc)
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
    def progress(self) -> str | None:
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

    def restart(self, kill: bool = False) -> None:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            raise ValueError(
                f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        call = SyftAPICall(
            server_uid=self.server_uid,
            path="job.restart",
            args=[],
            kwargs={"uid": self.id},
            blocking=True,
        )
        res = api.make_call(call)
        self.fetch()
        return res

    def kill(self) -> SyftError | SyftSuccess:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        call = SyftAPICall(
            server_uid=self.server_uid,
            path="job.kill",
            args=[],
            kwargs={"id": self.id},
            blocking=True,
        )
        res = api.make_call(call)
        self.fetch()
        return res

    def fetch(self) -> None:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            raise ValueError(
                f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        call = SyftAPICall(
            server_uid=self.server_uid,
            path="job.get",
            args=[],
            kwargs={"uid": self.id},
            blocking=True,
        )
        job: Job | None = api.make_call(call)
        if job is None:
            return None
        self.resolved = job.resolved
        if job.resolved:
            self.result = job.result

        self.status = job.status
        self.n_iters = job.n_iters
        self.current_iter = job.current_iter

    @property
    def subjobs(self) -> list["Job"] | SyftError:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        return api.services.job.get_subjobs(self.id)

    def get_subjobs(self, context: AuthedServiceContext) -> list["Job"] | SyftError:
        job_service = context.server.get_service("jobservice")
        return job_service.get_subjobs(context, self.id)

    @property
    def owner(self) -> UserView | SyftError:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        return api.services.user.get_current_user(self.id)

    def _get_log_objs(self) -> SyftLog | SyftError:
        api = APIRegistry.api_for(
            server_uid=self.server_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            raise ValueError(f"api is None. You must login to {self.server_uid}")
        return api.services.log.get(self.log_id)

    def logs(
        self, stdout: bool = True, stderr: bool = True, _print: bool = True
    ) -> str | None:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return (
                f"Can't access Syft API. You must login to {self.syft_server_location}"
            )

        has_permissions = True

        results = []
        if stdout:
            stdout_log = api.services.log.get_stdout(self.log_id)
            if isinstance(stdout_log, SyftError):
                results.append(f"Log {self.log_id} not available")
                has_permissions = False
            else:
                results.append(stdout_log)

        if stderr:
            try:
                stderr_log = api.services.log.get_stderr(self.log_id)
                if isinstance(stderr_log, SyftError):
                    results.append(f"Error log {self.log_id} not available")
                    has_permissions = False
                else:
                    results.append(stderr_log)
            except Exception:
                # no access
                if isinstance(self.result, Err):
                    results.append(self.result.value)
        else:
            # add short error
            if isinstance(self.result, Err):
                results.append(self.result.value)

        if has_permissions:
            has_storage_permission = api.services.log.has_storage_permission(
                self.log_id
            )
            if not has_storage_permission:
                prompt_warning_message(
                    message="This is a placeholder object, the real data lives on a different server and is not synced."
                )

        results_str = "\n".join(results)
        if not _print:
            return results_str
        else:
            print(results_str)
            return None

    # def __repr__(self) -> str:
    #     return f"<Job: {self.id}>: {self.status}"

    def status_badge(self) -> dict[str, str]:
        status = self.status
        if status in [JobStatus.COMPLETED]:
            badge_color = "label-green"
        elif status in [JobStatus.PROCESSING]:
            badge_color = "label-orange"
        elif status in [JobStatus.CREATED]:
            badge_color = "label-gray"
        elif status in [JobStatus.ERRORED, JobStatus.INTERRUPTED]:
            badge_color = "label-red"
        else:
            badge_color = "label-orange"
        return {"value": status.upper(), "type": badge_color}

    def summary_html(self) -> str:
        # TODO: Fix id for buttons
        # relative
        from ...util.notebook_ui.components.sync import CopyIDButton

        try:
            # type_html = f'<div class="label {self.type_badge_class()}">{self.object_type_name.upper()}</div>'
            job_name = self.user_code_name or self.endpoint or "Job"
            description_html = f"<span class='syncstate-description'>{job_name}</span>"
            worker_summary = ""
            if self.job_worker_id:
                worker_copy_button = CopyIDButton(
                    copy_text=str(self.job_worker_id), max_width=60
                )
                worker_summary = f"""
                <div style="display: table-row">
                    <span class='syncstate-col-footer'>{'on worker'}
                    {worker_copy_button.to_html()}</span>
                </div>
                """

            summary_html = f"""
                <div style="display: flex; gap: 8px; justify-content: start; width: 100%;">
                    {description_html}
                    <div style="display: flex; gap: 8px; justify-content: end; width: 100%;">
                        {CopyIDButton(copy_text=str(self.id), max_width=60).to_html()}
                    </div>
                </div>
                <div style="display: table-row">
                <span class='syncstate-col-footer'>{self.creation_time[:-7] if self.creation_time else ''}</span>
                </div>
                {worker_summary}
                """
            summary_html = summary_html.replace("\n", "")
        except Exception as e:
            print("Failed to build table", e)
            raise
        return summary_html

    def _coll_repr_(self) -> dict[str, Any]:
        # [Note]: Disable logs in table, to improve performance
        # logs = self.logs(_print=False, stderr=False)
        # if logs is not None:
        #     log_lines = logs.split("\n")
        # if len(log_lines) > 2:
        #     logs = f"... ({len(log_lines)} lines)\n" + "\n".join(log_lines[-2:])

        subjobs = self.subjobs

        def default_value(value: str) -> str:
            return value if value else "--"

        return {
            "Status": self.status_badge(),
            "Job": self.summary_html(),
            "# Subjobs": default_value(len(subjobs)),
            "Progress": default_value(self.progress),
            "ETA": default_value(self.eta_string),
        }

    @property
    def has_parent(self) -> bool:
        return self.parent_job_id is not None

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        _ = self.resolve
        logs = self.logs(_print=False)
        if logs is not None:
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

    @property
    def fetched_status(self) -> JobStatus:
        self.fetch()
        return self.status

    @property
    def requesting_user(self) -> UserView | SyftError:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        return api.services.user.view(self.requested_by)

    @property
    def server_name(self) -> str | SyftError | None:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        return api.server_name

    @property
    def parent(self) -> Self | SyftError:
        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_server_location}"
            )
        return api.services.job.get(self.parent_job_id)

    @property
    def ancestors_name_list(self) -> list[str] | SyftError:
        if self.parent_job_id:
            parent = self.parent
            if isinstance(parent, SyftError):
                return parent
            parent_name_list = parent.ancestors_name_list
            if isinstance(parent_name_list, SyftError):
                return parent_name_list
            parent_name_list.append(parent.user_code_name)
            return parent_name_list
        return []

    def _repr_html_(self) -> str:
        # relative
        from ...util.notebook_ui.components.sync import CopyIDButton

        identifier = random.randint(1, 2**32)  # nosec
        result_tab_id = f"Result_{identifier}"
        logs_tab_id = f"Logs_{identifier}"
        job_type = "JOB" if not self.parent_job_id else "SUBJOB"
        ancestor_name_list = self.ancestors_name_list
        if isinstance(ancestor_name_list, SyftError):
            return ancestor_name_list
        api_header = f"{self.server_name}/jobs/" + "/".join(ancestor_name_list)
        copy_id_button = CopyIDButton(copy_text=str(self.id), max_width=60)
        button_html = copy_id_button.to_html()
        creation_time = self.creation_time[:-7] if self.creation_time else "--"
        updated_at = str(self.updated_at)[:-7] if self.updated_at else "--"

        user_repr = "--"
        if self.requested_by and not isinstance(
            requesting_user := self.requesting_user, SyftError
        ):
            user_repr = f"{requesting_user.name} {requesting_user.email}"

        worker_attr = ""
        if self.job_worker_id:
            worker = self.worker
            if not isinstance(worker, SyftError):
                worker_pool_id_button = CopyIDButton(
                    copy_text=str(worker.worker_pool_name), max_width=60
                )
                worker_attr = f"""
                    <div style="margin-top: 6px; margin-bottom: 6px;">
                    <span style="font-weight: 700; line-weight: 19.6px; font-size: 14px; font: 'Open Sans'">
                        Worker Pool:</span>
                        {worker.name} on worker {worker_pool_id_button.to_html()}
                    </div>
                """

        logs = self.logs(_print=False)
        logs_lines = logs.strip().split("\n") if logs else []
        logs_lines.insert(0, "<strong>Message</strong>")

        logs_lines = [f"<code>{line}</code>" for line in logs_lines]
        logs_lines_html = "\n".join(logs_lines)

        template = Template(job_repr_template)
        return template.substitute(
            job_type=job_type,
            api_header=api_header,
            user_code_name=self.user_code_name,
            button_html=button_html,
            status=self.status.value.title(),
            creation_time=creation_time,
            updated_at=updated_at,
            worker_attr=worker_attr,
            no_subjobs=len(self.subjobs),
            logs_tab_id=logs_tab_id,
            result_tab_id=result_tab_id,
            identifier=identifier,
            logs_lines_html=logs_lines_html,
            result=self.result,
            user_repr=user_repr,
        )

    def wait(
        self, job_only: bool = False, timeout: int | None = None
    ) -> Any | SyftNotReady | SyftError:
        self.fetch()
        if self.resolved:
            return self.resolve

        api = APIRegistry.api_for(
            server_uid=self.syft_server_location,
            user_verify_key=self.syft_client_verify_key,
        )

        if api is None:
            raise ValueError(
                f"Can't access Syft API. You must login to server with id '{self.syft_server_location}'"
            )

        workers = api.services.worker.get_all()
        if not isinstance(workers, SyftError) and len(workers) == 0:
            return SyftError(
                message=f"Server {self.syft_server_location} has no workers. "
                f"You need to start a worker to run jobs "
                f"by setting n_consumers > 0."
            )

        print_warning = True
        counter = 0
        while True:
            self.fetch()
            if self.resolved:
                if isinstance(self.result, SyftError | Err) or self.status in [  # type: ignore[unreachable]
                    JobStatus.ERRORED,
                    JobStatus.INTERRUPTED,
                ]:
                    return self.result
                break
            if print_warning and self.result is not None:
                result_obj = api.services.action.get(  # type: ignore[unreachable]
                    self.result.id, resolve_nested=False
                )
                if isinstance(result_obj, SyftError | Err):
                    return result_obj
                if result_obj.is_link and job_only:  # type: ignore[unreachable]
                    print(
                        "You're trying to wait on a job that has a link as a result."
                        "This means that the job may be ready but the linked result may not."
                        "Use job.wait().get() instead to wait for the linked result."
                    )
                    print_warning = False

            sleep(1)

            if timeout is not None:
                counter += 1
                if counter > timeout:
                    return SyftError(message="Reached Timeout!")

        # if self.resolve returns self.result as error, then we
        # return SyftError and not wait for the result
        # otherwise if a job is resolved and not errored out, we wait for the result
        if not job_only and self.result is not None:  # type: ignore[unreachable]
            self.result.wait(timeout)

        return self.resolve  # type: ignore[unreachable]

    @property
    def resolve(self) -> Any | SyftNotReady:
        if not self.resolved:
            self.fetch()

        if self.resolved:
            return self.result
        return SyftNotReady(message=f"{self.id} not ready yet.")

    def get_sync_dependencies(self, context: AuthedServiceContext) -> list[UID]:  # type: ignore
        dependencies = []
        if self.result is not None and isinstance(self.result, ActionObject):
            dependencies.append(self.result.id.id)

        if self.log_id:
            dependencies.append(self.log_id)

        subjobs = self.get_subjobs(context)
        if isinstance(subjobs, SyftError):
            return subjobs

        subjob_ids = [subjob.id for subjob in subjobs]
        dependencies.extend(subjob_ids)

        if self.user_code_id is not None:
            dependencies.append(self.user_code_id)

        output = context.server.get_service("outputservice").get_by_job_id(  # type: ignore
            context, self.id
        )
        if isinstance(output, SyftError):
            return output
        elif output is not None:
            dependencies.append(output.id)

        return dependencies


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

    user_code_id: UID | None = None
    resolved: bool | None = None
    status: JobStatus | None = None
    n_iters: int | None = None
    current_iter: int | None = None
    creation_time: str | None = None

    result: ActionObject | None = None

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
    ) -> Self:
        info = cls(
            includes_metadata=metadata,
            includes_result=result,
            user_code_id=job.user_code_id,
        )

        if metadata:
            for attr in cls.__public_metadata_attrs__:
                setattr(info, attr, getattr(job, attr))

        if result:
            if not job.resolved:
                raise ValueError("Cannot sync result of unresolved job")
            if not isinstance(job.result, ActionObject):
                raise ValueError("Could not sync result of job")
            info.result = job.result

        return info


@instrument
@serializable(canonical_name="JobStashSQL", version=1)
class JobStash(ObjectStash[Job]):
    object_type = Job
    settings: PartitionSettings = PartitionSettings(
        name=Job.__canonical_name__, object_type=Job
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store)

    def set_result(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        add_permissions: list[ActionObjectPermission] | None = None,
    ) -> Result[Job | None, str]:
        if (
            isinstance(item.result, ActionObject)
            and item.result.syft_blob_storage_entry_id is not None
        ):
            item.result._clear_cache()

        return self.update(credentials, item, add_permissions)

    def get_active(self, credentials: SyftVerifyKey) -> Result[list[Job], str]:
        return self.get_all_by_field(
            credentials=credentials, field_name="status", field_value=JobStatus.CREATED
        )

    def get_by_worker(
        self, credentials: SyftVerifyKey, worker_id: str
    ) -> Result[list[Job], str]:
        return self.get_all_by_field(
            credentials=credentials, field_name="worker_id", field_value=worker_id
        )

    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Result[list[Job], str]:
        return self.get_all_by_field(
            credentials=credentials, field_name="user_code_id", field_value=user_code_id
        )

    def get_by_parent_id(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[list[Job], str]:
        return self.get_all_by_field(
            credentials=credentials, field_name="parent_job_id", field_value=uid
        )


@serializable()
class JobV1(SyncableSyftObject):
    __canonical_name__ = "JobItem"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    server_uid: UID
    result: Any | None = None
    resolved: bool = False
    status: JobStatus = JobStatus.CREATED
    log_id: UID | None = None
    parent_job_id: UID | None = None
    n_iters: int | None = 0
    current_iter: int | None = None
    creation_time: str | None = Field(
        default_factory=lambda: str(datetime.now(tz=timezone.utc))
    )
    action: Action | None = None
    job_pid: int | None = None
    job_worker_id: UID | None = None
    updated_at: DateTime | None = None
    user_code_id: UID | None = None
    requested_by: UID | None = None
    job_type: JobType = JobType.JOB


@migrate(JobV1, Job)
def migrate_job_update_v1_current() -> list[Callable]:
    return [
        make_set_default("endpoint", None),
    ]
