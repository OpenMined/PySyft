# stdlib
from typing import Any
from typing import Dict as TypeDict
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......core.common import UID
from ......core.common.serde.recursive import RecursiveSerde
from ......core.io.address import Address
from ......core.node.abstract.node import AbstractNode
from ......core.node.common.action.common import ImmediateSyftMessageWithoutReply
from ......core.node.common.node_table.dataset import Dataset


class DatasetCreateMessage(RecursiveSerde, ImmediateSyftMessageWithoutReply):  # type: ignore

    __attr_allowlist__ = ["address", "msg_id", "dataset", "_id"]

    def __init__(
        self,
        dataset: Dataset,
        address: Address,
        msg_id: Optional[UID] = None,
        **kwargs: TypeDict[Any, Any]
    ) -> None:
        super(ImmediateSyftMessageWithoutReply, self).__init__(
            address=address, msg_id=msg_id
        )
        self.dataset = dataset

    def process(
        self,
        node: AbstractNode,
        verify_key: VerifyKey,
    ) -> Any:
        print("got a datasetcreate message", self, type(node), type(verify_key))
