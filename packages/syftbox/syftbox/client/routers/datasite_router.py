# routers/datasite_router.py

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from syftbox.client.routers.common import APIContext

router = APIRouter()


# Don't think we require this Request model until we have
# an endpoint that allows one to create a datasite
class DatasiteRequest(BaseModel):
    name: str


@router.get("/")
async def list_datasites(ctx: APIContext) -> dict:
    """List all available datasites"""

    try:
        return {"datasites": ctx.all_datasites}
    except Exception as e:
        logger.error(f"Error listing datasites: {e}")
        raise HTTPException(status_code=500, detail="Failed to list datasites")
