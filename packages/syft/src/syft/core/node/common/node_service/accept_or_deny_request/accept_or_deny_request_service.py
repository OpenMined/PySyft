# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import debug
from ......logger import traceback_and_raise
from ......util import key_emoji
from ......util import validate_type
from .....common.message import ImmediateSyftMessageWithoutReply
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithoutReply
from .accept_or_deny_request_messages import AcceptOrDenyRequestMessage


class AcceptOrDenyRequestService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: ImmediateSyftMessageWithoutReply,
        verify_key: Optional[VerifyKey] = None,
    ) -> None:
        debug((f"> Processing AcceptOrDenyRequestService on {node.pprint}"))

        if verify_key is None:
            traceback_and_raise(
                ValueError(
                    "Can't process AcceptOrDenyRequestService without a "
                    "specified verification key"
                )
            )
        _msg: AcceptOrDenyRequestMessage = validate_type(
            msg, AcceptOrDenyRequestMessage
        )
        if _msg.accept:
            request_id = _msg.request_id
            for req in node.requests:
                if request_id == req.id:
                    # you must be a root user to accept a request
                    if verify_key == node.root_verify_key:
                        print("accepting reqest")
                        obj = node.store.get(req.object_id, proxy_only=True)
                        obj.read_permissions[req.requester_verify_key] = req.id

                        print(obj.read_permissions)

                        # gotta put the object back
                        node.store[req.object_id] = obj

                        print(
                            node.store.get(
                                req.object_id, proxy_only=True
                            ).read_permissions
                        )

                        node.requests.remove(req)

                        debug(f"> Accepting Request:{request_id} {request_id.emoji()}")
                        debug(
                            "> Adding can_read for 🔑 "
                            + f"{key_emoji(key=req.requester_verify_key)} to "
                            + f"Store UID {req.object_id} {req.object_id.emoji()}"
                        )
                        return None

        else:
            request_id = _msg.request_id
            for req in node.requests:
                if request_id == req.id:
                    # if you're a root user you can disable a request
                    # also people can disable their own requets
                    if (
                        verify_key == node.root_verify_key
                        or verify_key == req.requester_verify_key
                    ):
                        node.requests.remove(req)
                        debug(f"> Rejecting Request:{request_id}")
                        return None

    @staticmethod
    def message_handler_types() -> List[Type[AcceptOrDenyRequestMessage]]:
        return [AcceptOrDenyRequestMessage]
