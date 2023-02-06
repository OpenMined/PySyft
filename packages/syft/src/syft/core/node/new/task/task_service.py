# stdlib
from datetime import date
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

# third party
import requests
from result import Err
from result import Ok
from result import Result

# relative
from .....oblv.constants import LOCAL_MODE
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ...common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ...common.node_table.syft_object import SyftObject
from ..action_object import ActionObject
from ..action_store import ActionStore
from ..api import SyftAPI
from ..context import AuthedServiceContext
from ..document_store import DocumentStore
from ..service import AbstractService
from ..service import service_method
from .task import NodeView
from .task import Task
from .task_stash import TaskStash

DOMAIN_CONNECTION_PORT = int(os.getenv("DOMAIN_CONNECTION_PORT", 3030))


@serializable(recursive_serde=True)
class DictObject(SyftObject):
    # version
    __canonical_name__ = "Dict"
    __version__ = SYFT_OBJECT_VERSION_1

    base_dict: Dict[UID, dict] = {}

    # serde / storage rules
    __attr_state__ = ["id", "base_dict"]

    __attr_searchable__ = []
    __attr_unique__ = []


@serializable(recursive_serde=True)
class TaskService(AbstractService):
    document_store: DocumentStore
    task_stash: TaskStash
    action_store: ActionStore

    def __init__(
        self, document_store: DocumentStore, action_store: ActionStore
    ) -> None:
        self.document_store = document_store
        self.task_stash = TaskStash(store=document_store)
        self.action_store = action_store

    @service_method(path="task.create_enclave_task", name="create_enclave_task")
    def create_enclave_task(
        self,
        context: AuthedServiceContext,
        inputs: Dict[NodeView, dict],
        code: str,
        outputs: List[str],
        owners: List[NodeView],
        task_id: UID,
        oblv_metadata: Optional[dict] = None,
    ) -> Result[Ok, Err]:
        """Enclave Submit task"""
        # TODO 🟣 Check for permission after it is fully integrated

        if owners and not isinstance(owners, list):
            return Err(
                f"Enclave task creation should contain valid owners list : {owners}"
            )
        task = Task(
            id=task_id,
            inputs=inputs,
            code=code,
            outputs={var: " -- " for var in outputs},
            user=str(context.credentials),
            owners=owners,
            status={owner: "pending" for owner in owners},
            created_at=date.today().strftime("%d/%m/%Y"),
            updated_at=" -- ",
            reviewed_by=" -- ",
            execution="pending",
            oblv_metadata=oblv_metadata,
        )

        result = self.task_stash.set(task)

        if result.is_ok():
            return Ok(f"Added task to Enclave with id: {task_id}")

        return result.err()

    @service_method(path="task.create_domain_task", name="create_domain_task")
    def create_domain_task(
        self,
        context: AuthedServiceContext,
        inputs: Dict[NodeView, dict],
        code: str,
        outputs: List[str],
        owners: List[NodeView],
        task_id: UID,
        oblv_metadata: Optional[dict] = None,
    ) -> Result[Ok, Err]:
        """Enclave Submit task"""
        # TODO 🟣 Check for permission after it is fully integrated

        if owners and isinstance(owners, list) and len(owners) != 1:
            return Err("Domain task creation should have exactly one owner")
        task = Task(
            id=task_id,
            inputs=inputs,
            code=code,
            outputs={var: " -- " for var in outputs},
            user=str(context.credentials),
            owners=owners,
            status={owners[0]: "pending"},
            created_at=date.today().strftime("%d/%m/%Y"),
            updated_at=" -- ",
            reviewed_by=" -- ",
            execution="pending",
            oblv_metadata=oblv_metadata,
        )

        result = self.task_stash.set(task)

        if result.is_ok():
            return Ok(f"Added task to domain node: {owners[0].name}")

        return result.err()

    @service_method(path="task.get_task", name="get_task")
    def get_task(
        self,
        context: AuthedServiceContext,
        task_id: UID,
    ) -> Result[Ok, Err]:
        # TODO 🟣 Check for permission after it is fully integrated

        res = self.task_stash.get_by_uid(task_id)
        if res.is_ok():
            return res.ok()
        return res.err()

    @service_method(path="task.review_task", name="review_task")
    def review_task(
        self,
        context: AuthedServiceContext,
        task_id: UID,
        approve: bool,
        reason: str = "",
    ) -> Result[Ok, Err]:
        # TODO 🟣 Check for permission after it is fully integrated

        task = self.task_stash.get_by_uid(task_id)
        if task.is_ok():
            # 🟡 TODO: To Remove double nesting of result variable in Collection access
            task = task.ok().ok()
        else:
            return task.err()

        task.reason = reason

        # Fetch private data from action store if the code task is approved
        if approve:
            owner = task.owners[0]

            task.status[owner] = "Approved"
            # Retrive input map of the current domain
            private_input_map = {}
            owner_input_map = {}
            for node_view, input_map in task.inputs.items():
                if node_view == owner:
                    owner_input_map = input_map
                    break

            # Replace inputs with private data
            for input_data_name, input_id in owner_input_map.items():
                result = self.action_store.get(
                    uid=input_id, credentials=context.credentials
                )
                if result.is_ok():
                    private_input_map[input_data_name] = result.ok()
                else:
                    return Err(result.err())

        else:
            task.status[task.owners[0]] = "Denied"

        # Update task status back to DB
        res = self.task_stash.update(task)

        # If we are in the Enclave and have metadata for enclaves
        # Sent the task status to the connected enclaves
        if task.oblv_metadata:
            if LOCAL_MODE:
                api = self._get_api(
                    f"http://host.docker.internal:{DOMAIN_CONNECTION_PORT}"
                )
                enclave_res = api.services.task.send_status_to_enclave(
                    task, private_input_map
                )
                if isinstance(enclave_res, bool) and enclave_res:
                    return Ok(f"Sent task: {task_id} status to enclave")
                return Err(f"{enclave_res}")
            else:
                # TODO 🟣 Add code for real enclave deployment
                pass

        if res.is_ok():
            return Ok(f"Task: {task_id}  - {approve}")
        return res.err()

    @staticmethod
    def _get_api(connection_string: str) -> SyftAPI:

        req = requests.get(
            connection_string + "/worker/api",
        )
        obj = deserialize(req.content, from_bytes=True)
        obj.api_url = f"{connection_string}/worker/syft_api_call"
        return cast(SyftAPI, obj)

    @service_method(path="task.send_status_to_enclave", name="send_status_to_enclave")
    def send_status_to_enclave(
        self, context: AuthedServiceContext, task: Task, private_input_map: dict
    ) -> Result[Ok, Err]:
        enclave_task = self.task_stash.get_by_uid(task.id)
        if enclave_task.is_ok():
            # 🟡 TODO: To Remove double nesting of result variable in Collection access
            enclave_task = enclave_task.ok().ok()
        else:
            return enclave_task.err()

        task_owner = task.owners[0]
        if task_owner in enclave_task.owners:
            enclave_task.status[task_owner] = task.status[task_owner]
            # Create nested action store of DO for storing intermediate data
            if not self.action_store.exists(task_owner.node_uid):
                dict_object = DictObject()
                dict_object.base_dict[task.id] = private_input_map
                self.action_store.set(
                    uid=task_owner.node_uid,
                    credentials=context.credentials,
                    syft_object=dict_object,
                )
            else:
                result = self.action_store.get(
                    uid=task_owner.node_uid, credentials=context.credentials
                )
                if result.is_ok():
                    result = result.ok()
                    result.base_dict[task.id] = private_input_map
                else:
                    return result.err()

            return Ok(True)
        else:
            return Err(
                f"Task: {task.id.no_dash} in enclave does not contain {task.owners[0].name} "
            )

    @service_method(path="task.get", name="get")
    def get(self, context: AuthedServiceContext, uid: UID) -> Result[ActionObject, str]:
        """Get an object from the action store"""
        result = self.action_store.get(
            uid=uid, credentials=context.credentials, skip_permission=True
        )
        if result.is_ok():
            return Ok(result.ok())
        return Err(result.err())
