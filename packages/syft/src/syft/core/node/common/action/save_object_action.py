# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


@serializable(recursive_serde=True)
class SaveObjectAction(ImmediateActionWithoutReply):
    __attr_allowlist__ = ["obj", "id", "address"]

    def __init__(
        self,
        obj: StorableObject,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj = obj

    def __repr__(self) -> str:
        obj_str = str(self.obj)
        # make obj_str of reasonable length, if too long: cut into begin and end
        neg_index = max(-50, -len(obj_str) + 50)
        obj_str = (
            obj_str[:50]
            if len(obj_str) < 50
            else obj_str[:50] + " ... " + obj_str[neg_index:]
        )
        return f"SaveObjectAction {obj_str}"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        self.obj.read_permissions = {
            node.verify_key: node.id,
            verify_key: None,  # we dont have the passed in sender's UID
        }
        self.obj.write_permissions = {
            node.verify_key: node.id,
            verify_key: None,  # we dont have the passed in sender's UID
        }
        node.store[self.obj.id] = self.obj
