# stdlib
from typing import Any
from typing import Optional

# third party
from nacl.signing import SigningKey

# syft absolute
from syft.core.common.message import SyftMessage
from syft.core.io.address import Address
from syft.core.node.common.action.exception_action import ExceptionMessage
from syft.lib.python.dict import Dict

# grid absolute
from grid.core.node import node


def send_message_with_reply(
    signing_key: SigningKey,
    message_type: SyftMessage,
    address: Optional[Address] = None,
    reply_to: Optional[Address] = None,
    **content: Any
) -> Dict:
    if not address:
        address = node.address
    if not reply_to:
        reply_to = node.address
    # if flags.USE_NEW_SERVICE:
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
