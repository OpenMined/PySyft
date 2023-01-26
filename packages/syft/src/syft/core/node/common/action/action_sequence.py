# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .save_object_action import SaveObjectAction


@serializable(recursive_serde=True)
class ActionSequence(ImmediateActionWithoutReply):
    __attr_allowlist__ = ["obj_lst", "address", "id"]

    def __init__(
        self,
        obj_lst: List[SaveObjectAction],
        address: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_lst = obj_lst

    def __repr__(self) -> str:
        obj_str = str(self.obj_lst)
        # make obj_str of reasonable length, if too long: cut into begin and end
        neg_index = max(-50, -len(obj_str) + 50)
        obj_str = (
            obj_str[:50]
            if len(obj_str) < 50
            else obj_str[:50] + " ... " + obj_str[neg_index:]
        )
        return f"ActionSequence {obj_str}"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        for obj in self.obj_lst:
            obj.execute_action(node=node, verify_key=verify_key)
