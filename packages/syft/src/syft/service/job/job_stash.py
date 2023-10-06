# stdlib
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
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

    __attr_searchable__ = ["parent_job_id"]
    __repr_attrs__ = ["id", "result", "resolved"]

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

    @property
    def subjobs(self):
        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        return api.services.job.get_subjobs(self.id)

    def logs(self, _print=True):
        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        log_item = api.services.log.get(self.log_id)
        res = log_item.stdout
        if _print:
            print(res)
        else:
            return res

    # def __repr__(self) -> str:
    #     return f"<Job: {self.id}>: {self.status}"

    def _coll_repr_(self) -> Dict[str, Any]:
        logs = self.logs(_print=False)
        log_lines = logs.split("\n")
        subjobs = self.subjobs
        if len(log_lines) > 4:
            logs = "...\n" + "\n".join(log_lines[-4:])
        else:
            logs = logs

        if self.result is None:
            result = ""
        else:
            result = str(self.result.syft_action_data)

        return {
            "status": self.status,
            "logs": logs,
            "result": result,
            "has_parent": self.has_parent,
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
        if item.resolved:
            valid = self.check_type(item, self.object_type)
            if valid.is_err():
                return SyftError(message=valid.err())
            return super().update(credentials, item, add_permissions)
        return None

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
