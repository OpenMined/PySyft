# stdlib
from typing import Any
from typing import Dict

# third party
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/{domain_id}/{step}", status_code=200, response_class=JSONResponse)
def remote_ping(domain_id: str, step: int) -> Dict[str, Any]:
    if step == 0:
        return {"score": 0.95, "step": 0, "level": "pass", "total_steps": 4}
    elif step == 1:
        return {"score": 0.88, "step": 1, "level": "pass", "total_steps": 4}
    elif step == 2:
        return {"score": 0.99, "step": 2, "level": "pass", "total_steps": 4}
    elif step == 3:
        return {"score": 0.56, "step": 3, "level": "pass", "total_steps": 4}
    return {"name": domain_id, "step": step}
