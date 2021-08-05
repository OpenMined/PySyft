# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common import UID
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.serde.recursive import RecursiveSerde
from syft.core.io.address import Address
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.action.common import ImmediateActionWithoutReply
from syft.core.node.common.node_table.dataset import Dataset


class DatasetCreateMessage(ImmediateActionWithoutReply, RecursiveSerde):

    __attr_allowlist__ = ["address", "msg_id", "dataset"]

    def __init__(
        self, dataset: Dataset, address: Address, msg_id: Optional[UID] = None
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.dataset = dataset



# class DatasetManagerService(ImmediateNodeServiceWithReply):
#
#
#     @staticmethod
#     @service_auth(guests_welcome=True)
#     def process(
#         node: AbstractNode,
#         msg: INPUT_MESSAGES,
#         verify_key: VerifyKey,
#     ) -> OUTPUT_MESSAGES:
#         return DatasetManagerService.msg_handler_map[type(msg)](
#             msg=msg, node=node, verify_key=verify_key
#         )
#
#     @staticmethod
#     def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
#         return [
#         ]
