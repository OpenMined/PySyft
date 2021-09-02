# stdlib
from typing import List
from typing import NoReturn

# third party
from fastapi import APIRouter
from fastapi import Depends
from loguru import logger
from starlette import status
from starlette.exceptions import HTTPException

# grid absolute
from app.api import deps
from app.grid_requests.models import Request
from app.grid_requests.models import RequestUpdate
from app.users.models import UserPrivate

# relative
from . import syft as syft_requests_messages


def raise_generic_private_error() -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="There was an error processing your request.",
    )


router = APIRouter()


@router.get(
    "",
    response_model=List[Request],
    name="requests:read_all",
    status_code=status.HTTP_200_OK,
)
async def get_all_requests_grid(
    current_user: UserPrivate = Depends(deps.get_current_user),
) -> List[Request]:
    try:
        return syft_requests_messages.get_all_requests(current_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.get(
    "/{request_id}",
    response_model=Request,
    name="requests:read_one",
    status_code=status.HTTP_200_OK,
)
async def get_request_grid(
    request_id: str, current_user: UserPrivate = Depends(deps.get_current_user)
) -> Request:
    try:
        return syft_requests_messages.get_request(current_user, request_id)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.patch(
    "/{request_id}",
    name="requests:update",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def update_user_grid(
    request_id: str,
    updated_request: RequestUpdate,
    current_user: UserPrivate = Depends(deps.get_current_user),
) -> str:
    try:
        return syft_requests_messages.update_request(
            current_user, request_id, updated_request
        )
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()
