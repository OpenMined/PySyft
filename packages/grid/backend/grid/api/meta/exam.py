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


@router.get("/{domain_id}/{step}", status_code=200, response_class=JSONResponse)
def remote_ping(domain_id: str, step: str) -> Dict[str, Any]:
    if step == 0:
        return {"score": 0.95, "step": 0, "level": "pass", "total_steps": 4}
    if step == 1:
        return {"score": 0.88, "step": 1, "level": "pass", "total_steps": 4}
    if step == 2:
        return {"score": 0.99, "step": 2, "level": "pass", "total_steps": 4}
    if step == 3:
        return {"score": 0.56, "step": 3, "level": "pass", "total_steps": 4}
    return {"name": domain_id, "step": step}
