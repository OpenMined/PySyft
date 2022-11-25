# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# relative
from .....lib.python.list import List as SyftList
from .....proto.core.node.common.action.beaver_action_pb2 import (
    BeaverAction as BeaverAction_PB,
)
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serializable import serializable
from ....common.serde.serialize import _serialize as serialize
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
        values: Union[List[ShareTensor], List[str]],
        locations: List[UID],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.values = values
        self.locations = locations
        if len(values) != len(locations):
            raise ValueError(
                f"Iterable size for Values: {len(values)} Locations: {len(locations)} should be same for Beaver Action."
            )

    def __repr__(self) -> str:
        return (
            "Beaver Action: "
            + f"Values: {self.values}, "
            + f"Locations: {self.locations}"
        )

    @staticmethod
    def beaver_populate(
        data: Union[ShareTensor, str], id_at_location: UID, node: AbstractNode
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
            list_data = SyftList([data])
            result = StorableObject(
                id=id_at_location,
                data=list_data,
                read_permissions={},
            )
            BEAVER_CACHE[id_at_location] = result  # type: ignore
        elif isinstance(obj.data, SyftList):
            list_obj: SyftList = obj.data
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
        for value, location in zip(self.values, self.locations):
            BeaverAction.beaver_populate(value, location, node)  # type: ignore

    def _object2proto(self) -> BeaverAction_PB:
        values = [serialize(value, to_bytes=True) for value in self.values]
        locations = [serialize(location) for location in self.locations]
        addr = serialize(self.address)
        return BeaverAction_PB(values=values, locations=locations, address=addr)

    @staticmethod
    def _proto2object(proto: BeaverAction_PB) -> "BeaverAction":
        values = [deserialize(value, from_bytes=True) for value in proto.values]
        locations = [deserialize(location) for location in proto.locations]
        addr = deserialize(blob=proto.address)
        return BeaverAction(values=values, locations=locations, address=addr)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return BeaverAction_PB
