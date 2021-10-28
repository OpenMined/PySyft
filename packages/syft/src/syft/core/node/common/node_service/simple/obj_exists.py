# stdlib
from typing import Any
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ... import UID
from ....abstract.node import AbstractNode
from .simple_messages import NodeRunnableMessageWithReply


class DoesObjectExistMessage(NodeRunnableMessageWithReply):

    __attr_allowlist__ = ["obj_id"]

    def __init__(self, obj_id: UID):
        self.obj_id = obj_id

    def run(self, node: AbstractNode, verify_key: Optional[VerifyKey] = None) -> Any:

        try:
            node.store[self.obj_id]  # type: ignore
            return True
        except Exception:
            return False
