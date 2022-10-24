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
from syft.core.node.common.node_service.association_request.association_request_service import (
    get_vpn_status_metadata,
)
from syft.core.node.common.node_service.vpn.vpn_messages import (
    VPNConnectMessageWithReply,
)
from syft.core.node.common.node_service.vpn.vpn_messages import (
    VPNRegisterMessageWithReply,
)
from syft.core.node.common.node_service.vpn.vpn_messages import (
    VPNStatusMessageWithReply,
)
from syft.core.node.common.node_service.vpn.vpn_messages import VPNJoinMessageWithReply
from syft.grid import GridURL
from syft.lib.python.util import upcast

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.config import settings
from grid.core.node import node

router = APIRouter()


# this endpoint will tell the tailscale vpn container internally to connect to the
# supplied host with the supplied key
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
                "grid_url": GridURL.from_url(host_or_ip),
                "vpn_auth_key": vpn_auth_key,
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


# this endpoint will tell the node to contact the supplied host network and ask
# for a vpn key, then use that vpn key to connect
@router.post("/join/{host_or_ip}", status_code=200, response_class=JSONResponse)
def join(
    host_or_ip: str,
    current_user: Any = Depends(get_current_user),
) -> Dict[str, Any]:
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)
    msg = (
        VPNJoinMessageWithReply(kwargs={"grid_url": GridURL.from_url(host_or_ip)})
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


# this endpoint will ask the node to get the status of the vpn connection which returns
# a bool for connected, the host details and the peers
@router.get("/status", status_code=200, response_class=JSONResponse)
def status(
    current_user: Any = Depends(get_current_user),
) -> Dict[str, Any]:
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)
    msg = (
        VPNStatusMessageWithReply(kwargs={})
        .to(address=node.address, reply_to=node.address)
        .sign(signing_key=user_key)
    )
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    try:
        return upcast(reply.payload.kwargs)
    except Exception as e:
        print("failed", e, type(reply), type(reply.payload))
        pass

    return {"status": "error"}


# this endpoint will tell the network to create a vpn_auth_key for the domain to use
# to connect their tailscale vpn container to the networks headscale vpn key service
if settings.NODE_TYPE.lower() == "network":

    @router.post("/register", status_code=200, response_class=JSONResponse)
    def register() -> Dict[str, Any]:
        msg = (
            VPNRegisterMessageWithReply(kwargs={}).to(
                address=node.address, reply_to=node.address
            )
            # .sign(signing_key=user_key)
        )
        reply = node.recv_immediate_msg_with_reply(msg=msg).message

        result = {"status": "error"}
        try:
            # get the host_or_ip from tailscale
            try:
                vpn_metadata = get_vpn_status_metadata(node=node)
                result["host_or_ip"] = vpn_metadata["host_or_ip"]
            except Exception as e:
                print(f"failed to get get_vpn_status_metadata. {e}")
                result["host_or_ip"] = "100.64.0.1"
            result["node_id"] = str(node.target_id.id.no_dash)
            result["node_name"] = str(node.name)
            result["status"] = str(reply.payload.kwargs.get("status"))
            result["vpn_auth_key"] = str(reply.payload.kwargs.get("vpn_auth_key"))
        except Exception as e:
            print("failed", e, type(reply), type(reply.payload))

        return result
