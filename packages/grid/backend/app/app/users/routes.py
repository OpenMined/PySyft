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
from app.users.models import User
from app.users.models import UserCreate
from app.users.models import UserPrivate
from app.users.models import UserUpdate

# relative
from . import syft as syft_user_messages


def raise_generic_private_error() -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="There was an error processing your request.",
    )


router = APIRouter()


@router.get("/me", response_model=User, name="users:me", status_code=status.HTTP_200_OK)
def get_self(current_user: UserPrivate = Depends(deps.get_current_user)) -> User:
    return current_user


# TODO: Syft should return the newly created user and the response model should be User.
@router.post("", name="users:create", status_code=status.HTTP_201_CREATED)
async def create_user_grid(
    new_user: UserCreate,
    current_user: UserPrivate = Depends(deps.get_current_user),
) -> str:
    try:
        return syft_user_messages.create_user(new_user, current_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.get(
    "",
    response_model=List[User],
    name="users:read_all",
    status_code=status.HTTP_200_OK,
)
async def get_all_users_grid(
    current_user: UserPrivate = Depends(deps.get_current_user),
) -> List[User]:
    try:
        return syft_user_messages.get_all_users(current_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.get(
    "/{user_id}",
    response_model=User,
    name="users:read_one",
    status_code=status.HTTP_200_OK,
)
async def get_user_grid(
    user_id: int, current_user: UserPrivate = Depends(deps.get_current_user)
) -> User:
    try:
        return syft_user_messages.get_user(user_id, current_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.patch(
    "/{user_id}",
    name="users:update",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def update_user_grid(
    user_id: int,
    updated_user: UserUpdate,
    current_user: UserPrivate = Depends(deps.get_current_user),
) -> None:
    try:
        syft_user_messages.update_user(user_id, current_user, updated_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.delete(
    "/{user_id}", name="users:delete", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_user_grid(
    user_id: int, current_user: UserPrivate = Depends(deps.get_current_user)
) -> None:
    try:
        syft_user_messages.delete_user(user_id, current_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()
