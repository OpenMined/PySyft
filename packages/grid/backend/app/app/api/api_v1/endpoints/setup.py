from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
import json


from app.core.node import domain

router = APIRouter()


@router.get("", response_model=str)
async def syft_metadata(request: Request) -> Any:
    data = await request.body()
    return Response(
        json.dumps({}),
        media_type="application/json",
    )
