# stdlib
from typing import Any
from typing import Optional

# third party
from nacl.signing import SigningKey

# syft absolute
from syft.core.common.message import SyftMessage
from syft.core.common.uid import UID
from syft.core.node.common.action.exception_action import ExceptionMessage
from syft.lib.python.dict import Dict

# grid absolute
from grid.core.node import node


def send_message_with_reply(
    signing_key: SigningKey,
    message_type: SyftMessage,
    address: Optional[UID] = None,
    reply_to: Optional[UID] = None,
    **content: Any
) -> Dict:
    if not address:
        address = node.node_uid
    if not reply_to:
        reply_to = node.node_uid

    msg = message_type(address=address, reply_to=reply_to, kwargs=content).sign(
        signing_key=signing_key
    )

    reply = node.recv_immediate_msg_with_reply(msg=msg)
    reply = reply.message
    check_if_syft_reply_is_exception(reply)
    reply = reply.payload
    return reply


def check_if_syft_reply_is_exception(reply: Dict) -> None:
    if isinstance(reply, ExceptionMessage):
        raise Exception(reply.exception_msg)
