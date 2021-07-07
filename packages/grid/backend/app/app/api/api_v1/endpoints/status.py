from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("", status_code=200)
def ping() -> str:
    """
    Ping? Pong!
    """
    return "pong"


