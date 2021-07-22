# third party
from fastapi import APIRouter

router = APIRouter()


@router.get("", status_code=200)
def ping() -> str:
    """
    Ping? Pong!
    """
    return "pong"
