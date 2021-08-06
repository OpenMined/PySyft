# stdlib
from typing import Any
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......core.node.abstract.node import AbstractNode
from ......core.node.common.action.common import ImmediateSyftMessageWithoutReply
from ......core.node.common.node_service.node_service import (
    ImmediateNodeServiceWithReply,
)
from ..auth import service_auth
from .dataset_manager_messages import DatasetCreateMessage


class DatasetManagerService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Any,
        verify_key: VerifyKey,
    ) -> Any:
        allowed_messages = DatasetManagerService.message_handler_types()
        if type(msg) not in allowed_messages:
            raise Exception(
                f"Trying to process a {type(msg)} when only: {allowed_messages} allowed"
            )

        return msg.process(node=node, verify_key=verify_key)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithoutReply]]:
        return [DatasetCreateMessage]
