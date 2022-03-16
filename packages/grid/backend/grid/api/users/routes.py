# stdlib
import json
from typing import List
from typing import NoReturn
from typing import Optional

# third party
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from loguru import logger
from starlette import status
from starlette.exceptions import HTTPException

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.api.users.models import ApplicantStatus
from grid.api.users.models import User
from grid.api.users.models import UserCandidate
from grid.api.users.models import UserCreate
from grid.api.users.models import UserPrivate
from grid.api.users.models import UserUpdate

# relative
from . import syft as syft_user_messages


def raise_generic_private_error() -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="There was an error processing your request.",
    )


router = APIRouter()


@router.get("/me", response_model=User, name="users:me", status_code=status.HTTP_200_OK)
def get_self(current_user: UserPrivate = Depends(get_current_user)) -> User:
    return current_user


# TODO: Syft should return the newly created user and the response model should be User.
@router.post("", name="users:create", status_code=status.HTTP_201_CREATED)
async def create_user_grid(
    current_user: UserPrivate = Depends(get_current_user),
    new_user: str = Form(...),
    file: Optional[UploadFile] = File(None),
) -> str:
    if file:
        pdf_file = file.file.read()  # type: ignore
    else:
        pdf_file = b""

    dict_user = json.loads(new_user)
    dict_user["daa_pdf"] = pdf_file
    user_schema = UserCreate(**dict_user)
    try:
        return syft_user_messages.create_user(user_schema, current_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.get("/applicants", name="users:applicants", status_code=status.HTTP_201_CREATED)
async def get_all_candidates(
    current_user: UserPrivate = Depends(get_current_user),
) -> List[UserCandidate]:
    try:
        return syft_user_messages.get_user_requests(current_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.patch(
    "/applicants/{candidate_id}",
    name="users:applicants:process",
    status_code=status.HTTP_201_CREATED,
)
async def process_applicant_request(
    candidate_id: int,
    request_status: ApplicantStatus,
    current_user: UserPrivate = Depends(get_current_user),
) -> str:
    try:
        return syft_user_messages.process_applicant_request(
            current_user=current_user,
            candidate_id=candidate_id,
            status=request_status.status,
        )
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
    current_user: UserPrivate = Depends(get_current_user),
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
    user_id: int, current_user: UserPrivate = Depends(get_current_user)
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
    current_user: UserPrivate = Depends(get_current_user),
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
    user_id: int, current_user: UserPrivate = Depends(get_current_user)
) -> None:
    try:
        syft_user_messages.delete_user(user_id, current_user)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()
