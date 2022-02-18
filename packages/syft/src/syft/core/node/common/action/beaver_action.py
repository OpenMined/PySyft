# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# relative
from .....proto.core.node.common.action.beaver_action_pb2 import (
    BeaverAction as BeaverAction_PB,
)
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ....tensor.smpc.share_tensor import ShareTensor
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply

BEAVER_CACHE: Dict[UID, StorableObject] = {}  # Global cache for spdz mask values


@serializable()
class BeaverAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        eps: ShareTensor,
        eps_id: UID,
        delta: ShareTensor,
        delta_id: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.eps = eps
        self.eps_id = eps_id
        self.delta = delta
        self.delta_id = delta_id

    def __repr__(self) -> str:
        res = f"EPS: {self.eps}, "
        res = f"{res}EPS_ID: {self.eps_id}, "
        res = f"{res}DELTA: {self.delta}, "
        res = f"{res}DELTA_ID {self.delta_id} "
        return f"Beaver Action: {res}"

    @staticmethod
    def beaver_populate(
        data: ShareTensor, id_at_location: UID, node: AbstractNode
    ) -> None:
        """Populate the given input ShareTensor in the location specified.

        Args:
            data (Tensor): input ShareTensor to store in the node.
            id_at_location (UID): the location to store the data in.
            node Optional[AbstractNode] : The node on which the data is stored.
        """
        # TODO: Rasswanth Should modify storage to DB context,done
        # temporarily here to prevent race condition.

        obj = BEAVER_CACHE.get(id_at_location, None)  # type: ignore
        if obj is None:
            list_data = sy.lib.python.List([data])
            result = StorableObject(
                id=id_at_location,
                data=list_data,
                read_permissions={},
            )
            BEAVER_CACHE[id_at_location] = result  # type: ignore
        elif isinstance(obj.data, sy.lib.python.List):
            list_obj: List = obj.data
            list_obj.append(data)
            result = StorableObject(
                id=id_at_location,
                data=list_obj,
                read_permissions={},
            )
            BEAVER_CACHE[id_at_location] = result  # type: ignore
        else:
            raise Exception(f"Object at {id_at_location} should be a List or None")

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        eps = self.eps
        eps_id = self.eps_id
        delta = self.delta
        delta_id = self.delta_id
        BeaverAction.beaver_populate(eps, eps_id, node)
        BeaverAction.beaver_populate(delta, delta_id, node)

    def _object2proto(self) -> BeaverAction_PB:
        eps = sy.serialize(self.eps)
        eps_id = sy.serialize(self.eps_id)
        delta = sy.serialize(self.delta)
        delta_id = sy.serialize(self.delta_id)
        addr = sy.serialize(self.address)
        return BeaverAction_PB(
            eps=eps, eps_id=eps_id, delta=delta, delta_id=delta_id, address=addr
        )

    @staticmethod
    def _proto2object(proto: BeaverAction_PB) -> "BeaverAction":
        eps = sy.deserialize(blob=proto.eps)
        eps_id = sy.deserialize(blob=proto.eps_id)
        delta = sy.deserialize(blob=proto.delta)
        delta_id = sy.deserialize(blob=proto.delta_id)
        addr = sy.deserialize(blob=proto.address)
        return BeaverAction(
            eps=eps, eps_id=eps_id, delta=delta, delta_id=delta_id, address=addr
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return BeaverAction_PB
