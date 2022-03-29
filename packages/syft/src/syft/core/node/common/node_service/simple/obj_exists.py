# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ... import UID
from ......logger import info
from ....abstract.node import AbstractNode
from .simple_messages import NodeRunnableMessageWithReply


class DoesObjectExistMessage(NodeRunnableMessageWithReply):

    __attr_allowlist__ = ["obj_id"]

    def __init__(self, obj_id: UID) -> None:
        self.obj_id = obj_id

    def run(self, node: AbstractNode, verify_key: Optional[VerifyKey] = None) -> bool:
        try:
            return bool(node.store.get_or_none(self.obj_id, proxy_only=True))  # type: ignore
        except Exception as e:
            info("Exception in DoesObjectExistMessage:" + str(e))
            return False
