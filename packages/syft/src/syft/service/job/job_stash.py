# stdlib
from datetime import datetime
from datetime import timedelta
from enum import Enum
import random
from string import Template
from typing import Any

# third party
from pydantic import Field
from pydantic import model_validator
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ...client.api import APIRegistry
from ...client.api import SyftAPICall
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...service.context import AuthedServiceContext
from ...service.worker.worker_pool import SyftWorker
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_5
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.markdown import as_markdown_code
from ...util.telemetry import instrument
from ...util.util import prompt_warning_message
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..response import SyftError
from ..response import SyftNotReady
from ..response import SyftSuccess
from ..user.user import UserView
from .html_template import job_repr_template


@serializable()
class JobStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    ERRORED = "errored"
    COMPLETED = "completed"
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


@serializable()
class Job(SyncableSyftObject):
    __canonical_name__ = "JobItem"
    __version__ = SYFT_OBJECT_VERSION_5

    id: UID
    node_uid: UID
    result: Any | None = None
    resolved: bool = False
    status: JobStatus = JobStatus.CREATED
    log_id: UID | None = None
    parent_job_id: UID | None = None
    n_iters: int | None = 0
    current_iter: int | None = None
    creation_time: str | None = Field(default_factory=lambda: str(datetime.now()))
    action: Action | None = None
    job_pid: int | None = None
    job_worker_id: UID | None = None
    updated_at: DateTime | None = None
    user_code_id: UID | None = None
    requested_by: UID | None = None

    __attr_searchable__ = ["parent_job_id", "job_worker_id", "status", "user_code_id"]
    __repr_attrs__ = [
        "id",
        "result",
        "resolved",
        "progress",
        "creation_time",
        "user_code_name",
    ]
    __exclude_sync_diff_attrs__ = ["action", "node_uid"]
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
    def action_display_name(self) -> str:
        if self.action is None:
            return "action"
        else:
            # hacky
            self.action.syft_node_location = self.syft_node_location
            self.action.syft_client_verify_key = self.syft_client_verify_key
            return self.action.job_display_name

    @property
    def user_code_name(self) -> str | None:
        if self.user_code_id is not None:
            api = APIRegistry.api_for(
                node_uid=self.syft_node_location,
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
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_node_location}"
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

        def format_timedelta(local_timedelta: timedelta) -> str:
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
            if api is None:
                raise ValueError(
                    f"Can't access Syft API. You must login to {self.syft_node_location}"
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
        return None

    def kill(self) -> SyftError | None:
        if self.job_pid is not None:
            api = APIRegistry.api_for(
                node_uid=self.syft_node_location,
                user_verify_key=self.syft_client_verify_key,
            )
            if api is None:
                return SyftError(
                    message=f"Can't access Syft API. You must login to {self.syft_node_location}"
                )
            call = SyftAPICall(
                node_uid=self.node_uid,
                path="job.kill",
                args=[],
                kwargs={"id": self.id},
                blocking=True,
            )
            api.make_call(call)
            return None
        else:
            return SyftError(
                message="Job is not running or isn't running in multiprocessing mode."
            )

    def fetch(self) -> None:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            raise ValueError(
                f"Can't access Syft API. You must login to {self.syft_node_location}"
            )
        call = SyftAPICall(
            node_uid=self.node_uid,
            path="job.get",
            args=[],
            kwargs={"uid": self.id},
            blocking=True,
        )
        job: Job = api.make_call(call)
        self.resolved = job.resolved
        if job.resolved:
            self.result = job.result

        self.status = job.status
        self.n_iters = job.n_iters
        self.current_iter = job.current_iter

    @property
    def subjobs(self) -> list["Job"] | SyftError:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_node_location}"
            )
        return api.services.job.get_subjobs(self.id)

    def get_subjobs(self, context: AuthedServiceContext) -> list["Job"] | SyftError:
        job_service = context.node.get_service("jobservice")
        return job_service.get_subjobs(context, self.id)

    @property
    def owner(self) -> UserView | SyftError:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_node_location}"
            )
        return api.services.user.get_current_user(self.id)

    def _get_log_objs(self) -> SyftObject | SyftError:
        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            raise ValueError(f"api is None. You must login to {self.node_uid}")
        return api.services.log.get(self.log_id)

    def logs(
        self, stdout: bool = True, stderr: bool = True, _print: bool = True
    ) -> str | None:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return f"Can't access Syft API. You must login to {self.syft_node_location}"

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
                std_err_log = api.services.log.get_error(self.log_id)
                if isinstance(std_err_log, SyftError):
                    results.append(f"Error log {self.log_id} not available")
                    has_permissions = False
                else:
                    results.append(std_err_log)
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
                    message="This is a placeholder object, the real data lives on a different node and is not synced."
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
            description_html = (
                f"<span class='syncstate-description'>{self.user_code_name}</span>"
            )
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
        logs = self.logs(_print=False, stderr=False)
        if logs is not None:
            log_lines = logs.split("\n")
        subjobs = self.subjobs
        if len(log_lines) > 2:
            logs = f"... ({len(log_lines)} lines)\n" + "\n".join(log_lines[-2:])

        def default_value(value: str) -> str:
            return value if value else "--"

        return {
            "Status": self.status_badge(),
            "Job": self.summary_html(),
            "# Subjobs": center_content(default_value(len(subjobs))),
            "Progress": center_content(default_value(self.progress)),
            "ETA": center_content(default_value(self.eta_string)),
            "Logs": center_content(default_value(logs)),
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
    def requesting_user(self) -> UserView | SyftError:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_node_location}"
            )
        return api.services.user.view(self.requested_by)

    @property
    def node_name(self) -> str | SyftError | None:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_node_location}"
            )
        return api.node_name

    @property
    def parent(self) -> Self | SyftError:
        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            return SyftError(
                message=f"Can't access Syft API. You must login to {self.syft_node_location}"
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
        api_header = f"{self.node_name}/jobs/" + "/".join(ancestor_name_list)
        copy_id_button = CopyIDButton(copy_text=str(self.id), max_width=60)
        button_html = copy_id_button.to_html()
        creation_time = self.creation_time[:-7] if self.creation_time else "--"
        updated_at = str(self.updated_at)[:-7] if self.updated_at else "--"

        user_repr = "--"
        if self.requested_by:
            requesting_user = self.requesting_user
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
            uid=str(UID()),
            grid_template_columns=None,
            grid_template_cell_columns=None,
            cols=0,
            job_type=job_type,
            api_header=api_header,
            user_code_name=self.user_code_name,
            button_html=button_html,
            status=self.status.value.title(),
            creation_time=creation_time,
            user_rerp=user_repr,
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
    ) -> Any | SyftNotReady:
        # stdlib
        from time import sleep

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        if self.resolved:
            return self.resolve

        if not job_only and self.result is not None:
            self.result.wait(timeout)

        if api is None:
            raise ValueError(
                f"Can't access Syft API. You must login to {self.syft_node_location}"
            )
        print_warning = True
        counter = 0
        while True:
            self.fetch()
            if print_warning and self.result is not None:
                result_obj = api.services.action.get(
                    self.result.id, resolve_nested=False
                )
                if result_obj.is_link and job_only:
                    print(
                        "You're trying to wait on a job that has a link as a result."
                        "This means that the job may be ready but the linked result may not."
                        "Use job.wait().get() instead to wait for the linked result."
                    )
                    print_warning = False
            sleep(1)
            if self.resolved:
                break  # type: ignore[unreachable]
            # TODO: fix the mypy issue
            if timeout is not None:
                counter += 1
                if counter > timeout:
                    return SyftError(message="Reached Timeout!")
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

        output = context.node.get_service("outputservice").get_by_job_id(  # type: ignore
            context, self.id
        )
        if isinstance(output, SyftError):
            return output
        elif output is not None:
            dependencies.append(output.id)

        return dependencies


@serializable()
class JobInfo(SyftObject):
    __canonical_name__ = "JobInfo"
    __version__ = SYFT_OBJECT_VERSION_2

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
    ) -> Self:
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
            info.result = job.result

        return info


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
        add_permissions: list[ActionObjectPermission] | None = None,
    ) -> Result[Job | None, str]:
        valid = self.check_type(item, self.object_type)
        if valid.is_err():
            return SyftError(message=valid.err())
        return super().update(credentials, item, add_permissions)

    def get_by_result_id(
        self,
        credentials: SyftVerifyKey,
        res_id: UID,
    ) -> Result[Job | None, str]:
        res = self.get_all(credentials)
        if res.is_err():
            return res
        else:
            res = res.ok()
            # beautiful query
            res = [
                x
                for x in res
                if isinstance(x.result, ActionObject) and x.result.id.id == res_id
            ]
            if len(res) == 0:
                return Ok(None)
            elif len(res) > 1:
                return Err(SyftError(message="multiple Jobs found"))
            else:
                return Ok(res[0])

    def set_placeholder(
        self,
        credentials: SyftVerifyKey,
        item: Job,
        add_permissions: list[ActionObjectPermission] | None = None,
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
    ) -> Result[Job | None, str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        item = self.query_one(credentials=credentials, qks=qks)
        return item

    def get_by_parent_id(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Job | None, str]:
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

    def get_by_worker(
        self, credentials: SyftVerifyKey, worker_id: str
    ) -> Result[list[Job], str]:
        qks = QueryKeys(
            qks=[PartitionKey(key="job_worker_id", type_=str).with_obj(worker_id)]
        )
        return self.query_all(credentials=credentials, qks=qks)

    def get_by_user_code_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Result[list[Job], str]:
        qks = QueryKeys(
            qks=[PartitionKey(key="user_code_id", type_=UID).with_obj(user_code_id)]
        )

        return self.query_all(credentials=credentials, qks=qks)
