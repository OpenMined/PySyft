# stdlib
from typing import Dict

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....common.serde.serializable import serializable
from ..context import AuthedServiceContext
from ..document_store import DocumentStore
from ..service import AbstractService
from ..service import service_method
from .task_stash import TaskStash


@serializable(recursive_serde=True)
class TaskService(AbstractService):
    document_store: DocumentStore
    stash: TaskStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = TaskStash(store=store)

    @service_method(path="task.create_enclave_task", name="create_enclave_task")
    def create_enclave_task(
        self,
        context: AuthedServiceContext,
        inputs: Dict[str, str],
        code: str,
        outputs: Dict[str, str],
    ) -> Result[Ok, Err]:
        """Enclave Submit task"""
        # TODO 🟣 Check for permission after it is fully integrationed

        result = "Hello"
        return Ok(result)
