"""API for checking the status of the grid client."""

# third party
from fastapi import APIRouter

router = APIRouter()


@router.get("", status_code=200)
def ping() -> str:
    """Check if the grid client is up.

    Check for Ping? Get a Pong!

    Returns:
        str: pong
    """

    return "pong"
