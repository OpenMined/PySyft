# stdlib
from typing import Set as TypeSet
from uuid import UUID

# relative
from .....lib.python import Set
from ....common import UID
from ....common.message import SignedImmediateSyftMessageWithoutReply
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .exceptions import RetriableError

id = UID(UUID("a8ac0c37382584a1082c710b0b38f6a3"))


def get_set(node: AbstractNode) -> Set:
    obj = node.store.get_object(key=id)
    if isinstance(obj.data, Set):
        return obj.data
    elif obj is None:
        return Set([])
    else:
        raise ValueError(
            f"Unfinished task Object should be {obj} should be a Set or None"
        )


def update_set(
    task_set: TypeSet[SignedImmediateSyftMessageWithoutReply], node: AbstractNode
) -> None:
    obj = StorableObject(data=task_set, id=id)
    node.store[id] = obj


def register_unfinished_task(
    message: SignedImmediateSyftMessageWithoutReply, node: AbstractNode
) -> None:
    UNFINISHED_TASKS: TypeSet[SignedImmediateSyftMessageWithoutReply] = get_set(node)
    UNFINISHED_TASKS.add(message)
    update_set(UNFINISHED_TASKS, node)


def proceed_unfinished_tasks(node: AbstractNode) -> None:
    UNFINISHED_TASKS: TypeSet[SignedImmediateSyftMessageWithoutReply] = get_set(node)
    for unfinished_task in UNFINISHED_TASKS:
        try:
            node.recv_immediate_msg_without_reply(unfinished_task)
            UNFINISHED_TASKS.remove(unfinished_task)
        except Exception as e:
            print("Hello1")
            if isinstance(e, RetriableError):
                print("Hello2")
                print("Task Not Ready Yet")
            else:
                print("Hello3")
                raise e
    update_set(UNFINISHED_TASKS, node)
