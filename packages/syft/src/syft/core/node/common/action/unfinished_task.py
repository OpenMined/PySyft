# stdlib
from typing import Set

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.node.abstract.node import AbstractNode

# TODO: actually make the db work with all of this
UNFINISHED_TASKS: Set[SignedImmediateSyftMessageWithoutReply] = set()


def register_unfinished_task(message: SignedImmediateSyftMessageWithoutReply) -> None:
    UNFINISHED_TASKS.add(message)


def proceed_unfinished_tasks(node: AbstractNode) -> None:
    for unfinished_task in UNFINISHED_TASKS:
        node.recv_immediate_msg_without_reply(unfinished_task)
