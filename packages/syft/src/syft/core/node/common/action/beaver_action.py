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
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ....tensor.smpc.share_tensor import ShareTensor
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply

BEAVER_CACHE: Dict[UID, StorableObject] = {}  # Global cache for spdz mask values


@serializable(recursive_serde=True)
class BeaverAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        values: List[ShareTensor],
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
        for value, location in zip(self.values, self.locations):
            BeaverAction.beaver_populate(value, location, node)
