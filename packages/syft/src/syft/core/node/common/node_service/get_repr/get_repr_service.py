# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......util import traceback_and_raise
from .....common.group import VERIFYALL
from ....abstract.node import AbstractNode
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from .get_repr_messages import GetReprMessage
from .get_repr_messages import GetReprReplyMessage


class GetReprService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(root_only=True)
    def process(
        node: AbstractNode,
        msg: GetReprMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> GetReprReplyMessage:
        if verify_key is None:
            traceback_and_raise(
                "Can't process an GetReprService with no verification key."
            )

        obj = node.store.get(msg.id_at_location, proxy_only=True)
        contains_all_in_permissions = any(
            key is VERIFYALL for key in obj.read_permissions.keys()
        )

        if not (
            verify_key in obj.read_permissions.keys()
            or verify_key == node.root_verify_key
            or contains_all_in_permissions
        ):
            raise PermissionError("Permission to get repr of object not granted!")
        else:
            # TODO: Create a remote print interface for objects which displays them in a
            # nice way, we could also even buffer this between chained ops until we
            # return so that we can print once and display a nice list of data and ops
            # issue: https://github.com/OpenMined/PySyft/issues/5167
            result = repr(obj.data)
            return GetReprReplyMessage(repr=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[GetReprMessage]]:
        return [GetReprMessage]
