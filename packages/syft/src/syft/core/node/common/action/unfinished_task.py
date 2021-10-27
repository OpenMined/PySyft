# stdlib
from typing import Set
from uuid import UUID

# syft absolute
import syft as sy
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.node.common.action.exceptions import RetriableError
from syft.core.node.abstract.node import AbstractNode
from syft.core.common import UID
from syft.lib.python import Set

id = UID(UUID("a8ac0c37382584a1082c710b0b38f6a3"))

def get_set(node: AbstractNode) -> Set:
    obj = node.store.get_object(key=id)
    if isinstance(obj,Set):
        return obj
    elif obj is None:
        return Set([])
    else:
        raise ValueError(f"Unfinished task Object should be {obj} should be a Set or None")


def register_unfinished_task(message: SignedImmediateSyftMessageWithoutReply,node: AbstractNode) -> None:
    UNFINISHED_TASKS: Set[SignedImmediateSyftMessageWithoutReply] = get_set(node)
    UNFINISHED_TASKS.add(message)
    node.store[id] = UNFINISHED_TASKS


def proceed_unfinished_tasks(node: AbstractNode) -> None:
    UNFINISHED_TASKS: Set[SignedImmediateSyftMessageWithoutReply]
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
    node.store[id] = UNFINISHED_TASKS
