# stdlib
from typing import Any
from typing import Callable
from typing import Optional

# third party
import gevent
import gipc
from gipc.gipc import _GIPCDuplexHandle
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .document_store import BaseStash
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .service import AbstractService
from .service import service_method
from .user_stash import UserStash


@serializable(recursive_serde=True)
class Thing(SyftObject):
    # version
    __canonical_name__ = "Thing"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    number: float


@serializable(recursive_serde=True)
class ExecutorStash(BaseStash):
    object_type = Thing
    settings: PartitionSettings = PartitionSettings(
        name=Thing.__canonical_name__, object_type=Thing
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(self, thing: Thing) -> Result[Thing, str]:
        return super().set(thing)


def add_number_slowly(number: float) -> Thing:
    # third party
    import numpy as np

    x = np.array([range(1_000_000_00)])
    y = x.sum()
    thing = Thing(number=y)
    print("made thing", thing, thing.number, thing.id)
    return thing


class Task:
    func_bytes: bytes

    def __init__(self, func: Callable) -> None:
        # self.func_bytes = cloudpickle.dumps(func)
        pass

    @property
    def func(self) -> Callable:
        # return cloudpickle.loads(self.func_bytes)
        pass


def task_runner(pipe: _GIPCDuplexHandle) -> None:
    try:
        with pipe:
            task = pipe.get()
            print("got task", type(task))
            result = task.func()
            print("got result", result)
            pipe.put(result)
            pipe.close()
    except Exception as e:
        print("Exception in task_runner", e)
        pipe.close()
    print("Shutting down Task Process")


def task_producer(pipe: _GIPCDuplexHandle, task: Task) -> Any:
    print("Producer with pipe", pipe, "and task", task)
    try:
        with pipe:
            pipe.put(task)
            gevent.sleep(0)
            result = pipe.get()
            pipe.close()
            print("Producer has result:", result)
            return result
    except Exception as e:
        print("Exception in task_producer", e)
        pipe.close()
    print("Failed to produce result")


def queue_task(task: Task, save_result: Callable) -> Optional[Any]:
    with gipc.pipe(duplex=True) as (cend, pend):
        process = gipc.start_process(task_runner, args=(cend,))
        producer = gevent.spawn(task_producer, pend, task)
        try:
            process.join()
        except KeyboardInterrupt:
            producer.kill(block=True)
            process.terminate()
        process.join()

    result = producer.value
    save_result(result)
    print("Queue task got result", result)
    return result


@serializable(recursive_serde=True)
class ExecutorService(AbstractService):
    store: DocumentStore
    stash: UserStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ExecutorStash(store=store)

    @service_method(path="executor.run_rask", name="run_task")
    def run_task(
        self, context: AuthedServiceContext, number: float, blocking: bool = False
    ) -> Result[Ok, Err]:
        """Initial testing service"""

        def run_me():
            return add_number_slowly(number)

        def save_result(result):
            self.stash.set(result)

        task = Task(func=run_me)

        thread = gevent.spawn(queue_task, task, save_result)
        if blocking:
            gevent.joinall([thread])
            return Ok(thread.value)
        else:
            return Ok("Queued")

    @service_method(path="executor.get_results", name="get_results")
    def get_results(self, context: AuthedServiceContext) -> Result[Ok, Err]:
        return self.stash.get_all()
