# stdlib
from typing import List as TypeList
from uuid import UUID

# relative
from .....lib.python import List
from ....common import UID
from ....common.message import SignedImmediateSyftMessageWithoutReply
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .exceptions import RetriableError

id = UID(UUID("a8ac0c37382584a1082c710b0b38f6a3"))


def get_list(node: AbstractNode) -> TypeList[SignedImmediateSyftMessageWithoutReply]:
    obj = node.store.get_object(key=id)
    if obj is None:
        return List()  # type: ignore
    elif isinstance(obj.data, List):
        return obj.data  # type: ignore
    else:
        raise ValueError(
            f"Unfinished task Object should be {obj} should be a List or None"
        )


def update_list(
    task_list: TypeList[SignedImmediateSyftMessageWithoutReply], node: AbstractNode
) -> None:
    obj = StorableObject(data=task_list, id=id)
    node.store[id] = obj


def register_unfinished_task(
    message: SignedImmediateSyftMessageWithoutReply, node: AbstractNode
) -> None:
    UNFINISHED_TASKS: TypeList[SignedImmediateSyftMessageWithoutReply] = get_list(node)
    UNFINISHED_TASKS.append(message)
    update_list(UNFINISHED_TASKS, node)


def proceed_unfinished_tasks(node: AbstractNode) -> None:
    UNFINISHED_TASKS: TypeList[SignedImmediateSyftMessageWithoutReply] = get_list(node)
    for unfinished_task in UNFINISHED_TASKS:
        try:
            node.recv_immediate_msg_without_reply(unfinished_task)
            UNFINISHED_TASKS.remove(unfinished_task)
        except Exception as e:
            if not isinstance(e, RetriableError):
                raise e
    update_list(UNFINISHED_TASKS, node)
