# stdlib
from typing import Any
from typing import Dict

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi.responses import JSONResponse
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft absolute
from syft.core.node.common.node_service.vpn.vpn_messages import (
    VPNConnectMessageWithReply,
)

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.node import node

router = APIRouter()


@router.post("/connect/{host_or_ip}", status_code=200, response_class=JSONResponse)
def connect(
    host_or_ip: str,
    vpn_auth_key: str = Body(..., example="headscale vpn auth key"),
    network_id: str = Body(..., example="network UID"),
    current_user: Any = Depends(get_current_user),
) -> Dict[str, Any]:
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)
    msg = (
        VPNConnectMessageWithReply(
            kwargs={
                "host_or_ip": host_or_ip,
                "vpn_auth_key": vpn_auth_key,
                "network_id": network_id,
            }
        )
        .to(address=node.address, reply_to=node.address)
        .sign(signing_key=user_key)
    )
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    status = "error"
    try:
        status = str(reply.payload.kwargs.get("status"))
    except Exception as e:
        print("failed", e, type(reply), type(reply.payload))
        pass

    return {"status": status}
