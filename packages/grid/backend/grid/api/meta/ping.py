# stdlib
from typing import Any
from typing import Dict

# third party
from fastapi import APIRouter
from fastapi import Depends
from fastapi.responses import JSONResponse
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft absolute
from syft.core.node.common.node_service.ping.ping_messages import PingMessageWithReply
from syft.grid import GridURL

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.node import node

router = APIRouter()


@router.get("/{host_or_ip}", status_code=200, response_class=JSONResponse)
def remote_ping(
    host_or_ip: str, current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = (
        PingMessageWithReply(kwargs={"grid_url": GridURL.from_url(host_or_ip)})
        .to(address=node.address, reply_to=node.address)
        .sign(signing_key=user_key)
    )

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    return reply.payload
