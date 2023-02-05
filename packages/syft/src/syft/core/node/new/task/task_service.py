# stdlib
from datetime import date
from typing import Dict
from typing import List
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ..context import AuthedServiceContext
from ..document_store import DocumentStore
from ..service import AbstractService
from ..service import service_method
from .task import NodeView
from .task import Task
from .task_stash import TaskStash


@serializable(recursive_serde=True)
class TaskService(AbstractService):
    document_store: DocumentStore
    task_stash: TaskStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.task_stash = TaskStash(store=store)

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
        task.status[task.owners[0]] = approve
        res = self.task_stash.update(task)
        if res.is_ok():
            return Ok(f"Task: {task_id}  - {approve}")
        return res.err()
