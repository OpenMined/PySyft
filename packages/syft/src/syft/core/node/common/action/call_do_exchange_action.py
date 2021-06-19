# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ..... import serialize
from .....proto.core.node.common.action.call_do_exchange_pb2 import (
    CallDoExchangeAction as CallDoExchangeAction_PB,
)
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import Serializable
from ....common.serde.serializable import bind_protobuf
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply

@bind_protobuf
class CallDoExchangeAction(ImmediateActionWithoutReply, Serializable):
    def __init__(
        self,
        obj_id: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id

    def __repr__(self) -> str:
        obj_str = str(self.obj_id)
        # make obj_str of reasonable length, if too long: cut into begin and end
        neg_index = max(-50, -len(obj_str) + 50)
        obj_str = obj_str = (
            obj_str[:50]
            if len(obj_str) < 50
            else obj_str[:50] + " ... " + obj_str[neg_index:]
        )
        return f"SaveObjectAction {obj_str}"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        print(self.obj_id)
        obj = node.flight_client.get_object(self.obj_id).to_pandas()[str(self.obj_id.value)].to_numpy()
        print(obj)
        #TODO [IMP] (flight): adding object into store with permissions
        # node.store[str(self.obj_id.value)] = obj

    def _object2proto(self) -> CallDoExchangeAction_PB:
        # obj_name = self.obj_name._object2proto()
        addr = serialize(self.address)
        return CallDoExchangeAction_PB(obj_id=serialize(self.obj_id), address=addr)

    @staticmethod
    def _proto2object(proto: CallDoExchangeAction_PB) -> "CallDoExchangeAction":
        obj_id = _deserialize(blob=proto.obj_id)
        addr = _deserialize(blob=proto.address)
        return CallDoExchangeAction(obj_id=obj_id, address=addr)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return CallDoExchangeAction_PB
